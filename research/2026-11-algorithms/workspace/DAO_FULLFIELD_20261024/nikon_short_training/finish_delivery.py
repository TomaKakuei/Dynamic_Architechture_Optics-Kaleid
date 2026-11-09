"""Serialize label-free inference verification after ongoing confirmations."""
from pathlib import Path
import argparse,ctypes,json,os,subprocess,sys,traceback
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from durable_io import atomic_json

p=argparse.ArgumentParser();p.add_argument('--parent-pid',type=int,required=True);a=p.parse_args()
def state(status,**details):
    atomic_json(HERE/'delivery_state.json',dict(status=status,pid=os.getpid(),
        updated=datetime.now(timezone.utc).isoformat(),**details))
kernel=ctypes.windll.kernel32
kernel.OpenProcess.restype=ctypes.c_void_p
kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
kernel.CloseHandle.argtypes=[ctypes.c_void_p]
handle=kernel.OpenProcess(0x100000|0x1000,False,a.parent_pid)
kernel.SetThreadExecutionState(0x80000001)
try:
    if handle:
        state('waiting_for_confirmation',parent_pid=a.parent_pid)
        while True:
            result=kernel.WaitForSingleObject(handle,1000)
            if result==0:break
            if result!=258:raise RuntimeError(f'Process wait failed:{result}')
    assert json.loads((HERE/'exact_completion_state.json').read_text())['status']=='completed'
    for script in ('verify_inference.py','write_review.py'):
        state('running',script=script)
        subprocess.run([sys.executable,'-u',str(HERE/script)],check=True,cwd=HERE.parent.parent,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    state('completed',assistant_final_review='pending')
except Exception:
    state('failed',error=traceback.format_exc());raise
finally:
    if handle:kernel.CloseHandle(handle)
    kernel.SetThreadExecutionState(0x80000000)
