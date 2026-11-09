"""Attach to the already-running repair pilot, then run authorized training."""
from pathlib import Path
import ctypes,json,sys,subprocess,datetime,os
HERE=Path(__file__).resolve().parent


def record(stage,**kw):
    data=dict(stage=stage,pid=os.getpid(),updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
    (HERE/'coordinator_state.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(data),flush=True)


def main():
    pilot_pid=int(sys.argv[1]);record('waiting_existing_pilot',pilot_pid=pilot_pid)
    kernel=ctypes.windll.kernel32
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x00100000,False,pilot_pid)
    if handle:
        kernel.WaitForSingleObject(handle,0xFFFFFFFF);kernel.CloseHandle(handle)
    result=HERE/'repair_pilot/result.json'
    if not result.exists():record('stopped_pilot_incomplete');return
    pilot=json.loads(result.read_text())
    if not pilot['gate_pass']:record('stopped_pilot_gate_failed');return
    record('starting_four_epoch_training')
    with (HERE/'training.stdout.log').open('a',encoding='utf-8') as stdout,(HERE/'training.stderr.log').open('a',encoding='utf-8') as stderr:
        process=subprocess.Popen([sys.executable,'-X','utf8','-u','train4.py'],cwd=HERE,stdout=stdout,stderr=stderr)
        record('training_running',training_pid=process.pid)
        code=process.wait()
    record('finished' if code==0 else 'training_failed',exit_code=code,assistant_reviewed=False)


if __name__=='__main__':main()
