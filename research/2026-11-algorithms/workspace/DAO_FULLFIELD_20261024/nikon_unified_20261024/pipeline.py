"""Durable serial1x then2x four-epoch runs, followed by complete comparisons."""
from pathlib import Path
import sys,os,json,subprocess,traceback,ctypes
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from durable_io import atomic_json
def stamp():return datetime.now(timezone.utc).isoformat()
def state(status,**extra):
    atomic_json(HERE/'state.json',dict(status=status,parent_pid=os.getpid(),updated=stamp(),**extra))
def run(script,args,stage):
    cmd=[sys.executable,'-u',str(HERE/script),*args]
    with (HERE/(stage+'.log')).open('a',encoding='utf-8') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=HERE.parent.parent,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        state(stage,child_pid=child.pid,command=cmd)
        code=child.wait()
    if code:raise RuntimeError(f'{stage} failed with exit{code}; see{stage}.log; no automatic restart')

def main():
    lock=(HERE/'pipeline.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        assert json.loads((HERE/'preflight.json').read_text())['passed']
        for size in ('1x','2x'):
            if (HERE/('unified_'+size)/'training_evidence.json').exists():continue
            run('train.py',['--size',size],'training_'+size)
        if not (HERE/'comparison.csv').exists():run('postprocess.py',[],'postprocessing')
        state('completed',assistant_review='pending')
        current='CURRENT '+stamp()+'\n两个统一网络4轮训练及全量对照计算完成，等待助手核验。\n查看RESULTS.txt、comparison.json和comparison.csv。Trepan与原权重未变。\n\n'
        path=HERE/'HANDOFF.txt';old=path.read_text(encoding='utf-8') if path.exists() else ''
        path.write_text(current+'--- Previous handoff ---\n'+old,encoding='utf-8')
    except Exception:
        state('failed',error=traceback.format_exc());raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()

if __name__=='__main__':main()
