"""Wait for the existing evaluator, then analyze saved results on CPU only."""
import argparse,ctypes,json,os,subprocess,sys,datetime,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent;FULL=ROOT.parent/'olympus_full500_20261027'
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32];kernel.OpenProcess.restype=ctypes.c_void_p
kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];kernel.CloseHandle.argtypes=[ctypes.c_void_p]


def state(stage,**kw):
    p=ROOT/'watch_state.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2)+'\n',encoding='utf-8')
    os.replace(tmp,p)


def main(pid):
    state('waiting_for_existing_full500',evaluator_pid=pid,gpu_work=False)
    if not (FULL/'result.json').exists():
        handle=kernel.OpenProcess(0x00100000,False,pid)
        if not handle:raise OSError(ctypes.get_last_error(),'Cannot attach existing evaluator')
        try:
            while kernel.WaitForSingleObject(handle,5000)==258:
                if (FULL/'STOP_REQUESTED.json').exists():state('stopped_with_evaluator');return
        finally:kernel.CloseHandle(handle)
    if not (FULL/'result.json').exists():state('evaluation_incomplete_no_followup');return
    state('cpu_analysis',gpu_work=False)
    with (ROOT/'final_diagnosis.stdout.log').open('w',encoding='utf-8') as out,(ROOT/'final_diagnosis.stderr.log').open('w',encoding='utf-8') as err:
        completed=subprocess.run([sys.executable,'-X','utf8','analyze_saved.py'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    if completed.returncode:raise RuntimeError('Saved-result analysis failed')
    state('diagnostic_results_generated',assistant_interpretation_pending=True,automatic_gpu_work=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attach',type=int,required=True)
    try:main(p.parse_args().attach)
    except BaseException:state('failed',error=traceback.format_exc());raise
