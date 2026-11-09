"""Bounded serial background job: missing observations -> joint training -> summary."""
from pathlib import Path
import os, sys, json, time, ctypes
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
os.environ['PYTHONUTF8']='1'
def stamp():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path,value):
    temporary=Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for attempt in range(20):
        try:os.replace(temporary,path);return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.1)
def state(stage,**kw):atomic(HERE/'state.json',dict(stage=stage,updated=stamp(),pid=os.getpid(),**kw))
import subprocess, traceback, psutil

def main():
    (HERE/'logs').mkdir(exist_ok=True)
    owner=dict(pid=os.getpid(),created=psutil.Process().create_time(),started=stamp(),command=sys.argv,
               notifications_configured=False,independent_review=False)
    atomic(HERE/'started.json',owner)
    if os.name=='nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        stages=[('data',[sys.executable,'-u','data.py','--workers','8'])]
        if not (HERE/'training/warm_start.pt').exists() and not (HERE/'training/resume.pt').exists():
            stages.append(('startup',[sys.executable,'-u','train.py','startup']))
        stages.append(('train',[sys.executable,'-u','train.py','train']))
        for name,command in stages:
            print(stamp(), 'START',name,flush=True)
            with (HERE/'logs'/f'{name}.log').open('a',encoding='utf-8') as out:
                process=subprocess.Popen(command,cwd=HERE,stdout=out,stderr=subprocess.STDOUT)
                atomic(HERE/'active_stage.json',dict(stage=name,pid=process.pid,started=stamp()))
                result=process.wait()
            if result:raise RuntimeError(f'{name} exited {result}; see logs/{name}.log')
            print(stamp(),'FINISH',name,flush=True)
        summary=read(HERE/'training/summary.json')
        atomic(HERE/'complete.json',dict(completed=stamp(),training=summary,independent_review=False,
                                       background_running=False,remaining_est_seconds=0))
        (HERE/'HANDOFF.txt').write_text(
            'COMPLETE: Olympus Mod. joint training finished.\n'
            'Results: training/summary.json; selected candidate: training/candidate.pt.\n'
            'Curves: training/curves.csv, curves.png, curves.pdf.\n'
            'Only normal training validation was run. No independent replay or digest comparisons.\n'
            'No automatic deployment or notification. Original data/models preserved.\n',encoding='utf-8')
    except BaseException:
        error=traceback.format_exc()
        atomic(HERE/'failure.json',dict(time=stamp(),error=error,resume='Run pipeline.py; committed data blocks and training checkpoint are reused.'))
        state('failed',background_running=False,error=error)
        print(error,flush=True);raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()

