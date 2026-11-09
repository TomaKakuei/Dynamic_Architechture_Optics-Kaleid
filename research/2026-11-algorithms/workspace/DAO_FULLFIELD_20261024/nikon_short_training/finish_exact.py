"""Wait on the actual training parent handle, then finish full confirmations."""
from pathlib import Path
import argparse,ctypes,json,os,subprocess,sys,traceback
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from durable_io import atomic_json

def stamp():return datetime.now(timezone.utc).isoformat()
def state(status,**details):
    atomic_json(HERE/'exact_completion_state.json',dict(status=status,pid=os.getpid(),updated=stamp(),**details))

p=argparse.ArgumentParser();p.add_argument('--parent-pid',type=int,required=True);a=p.parse_args()
kernel=ctypes.windll.kernel32
kernel.OpenProcess.restype=ctypes.c_void_p
kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
kernel.CloseHandle.argtypes=[ctypes.c_void_p]
handle=kernel.OpenProcess(0x100000|0x1000,False,a.parent_pid)
kernel.SetThreadExecutionState(0x80000001)
try:
    if handle:
        state('waiting_for_training_parent',parent_pid=a.parent_pid)
        while True:
            result=kernel.WaitForSingleObject(handle,1000)
            if result==0:break
            if result!=258:raise RuntimeError(f'Process wait failed:{result}')
    pipeline=json.loads((HERE/'state.json').read_text())
    if pipeline['status']!='completed':raise RuntimeError(f'Training pipeline terminal status:{pipeline}')
    for noise in ('original','new'):
        directory='original_protocol_comparison' if noise=='original' else 'noise_confirmation'
        if (HERE/'exact_local_response'/directory/'summary.json').exists():continue
        cmd=[sys.executable,'-u',str(HERE/'confirm_noise.py'),'--idea','exact_local_response','--noise',noise]
        with (HERE/f'exact_confirmation_{noise}.log').open('a',encoding='utf-8') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=HERE.parent.parent,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state('confirming',noise=noise,child_pid=child.pid,command=cmd)
            code=child.wait()
        if code:raise RuntimeError(f'{noise} confirmation failed:{code}; saved results retained')
    state('completed',assistant_review='pending')
except Exception:
    state('failed',error=traceback.format_exc());raise
finally:
    if handle:kernel.CloseHandle(handle)
    kernel.SetThreadExecutionState(0x80000000)
