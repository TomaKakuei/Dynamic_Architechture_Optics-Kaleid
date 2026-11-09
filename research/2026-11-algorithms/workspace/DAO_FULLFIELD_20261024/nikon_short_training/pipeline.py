"""Serial short training; each declared idea gets exactly4full epochs."""
from pathlib import Path
import sys,os,json,subprocess,traceback,ctypes
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from durable_io import atomic_json
from datetime import datetime,timezone
def stamp():return datetime.now(timezone.utc).isoformat()
def summarize():
    results={}
    for idea in json.loads((HERE/'plan.json').read_text())['jobs']:
        folder=HERE/idea
        if not (folder/'summary.json').exists():continue
        summary=json.loads((folder/'summary.json').read_text())
        history=json.loads((folder/'history.json').read_text())
        evidence=json.loads((folder/'training_evidence.json').read_text())
        config=json.loads((folder/'config.json').read_text())
        results[idea]=dict(inference_protocol=config.get('inference_protocol','one_shot'),selected_epoch=summary['selected_epoch'],best=summary['validation'],
             epoch4=history[-1]['validation'],baseline=summary['baseline'],training=evidence,
             active_seconds=summary['active_seconds'])
    atomic_json(HERE/'comparison.json',dict(results=results,updated=stamp(),assistant_review='pending'))

def main():
    lock=(HERE/'pipeline.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for idea in json.loads((HERE/'plan.json').read_text())['jobs']:
            done=(HERE/idea/'training_evidence.json').exists()
            if idea=='closed_loop_training':done=done and (HERE/idea/'one_shot_summary.json').exists()
            if done:continue
            cmd=[sys.executable,'-u',str(HERE/'short_train.py'),'--idea',idea]
            with (HERE/f'{idea}.log').open('a',encoding='utf-8') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=HERE.parent.parent,
                                       creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                atomic_json(HERE/'state.json',dict(status='training',idea=idea,parent_pid=os.getpid(),child_pid=child.pid,
                    epochs_per_idea=4,updated=stamp(),command=cmd))
                code=child.wait()
            if code:raise RuntimeError(f'{idea} failed:exit{code}; see log; no automatic retry')
            summarize()
        if (HERE/'nonlinear_projection/training_evidence.json').exists():
            for idea in json.loads((HERE/'plan.json').read_text()).get('loop_jobs',['bounded_control','nonlinear_projection']):
                if (HERE/idea/'loop_summary.json').exists():continue
                cmd=[sys.executable,'-u',str(HERE/'evaluate_loop.py'),'--idea',idea]
                with (HERE/f'{idea}_loop.log').open('a',encoding='utf-8') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=HERE.parent.parent,
                        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    atomic_json(HERE/'state.json',dict(status='evaluating_loop',idea=idea,parent_pid=os.getpid(),
                        child_pid=child.pid,updated=stamp(),command=cmd))
                    code=child.wait()
                if code:raise RuntimeError(f'{idea} loop failed:exit{code}; training results retained')
        if json.loads((HERE/'plan.json').read_text()).get('post_training_diagnostics',False):
            cmd=[sys.executable,'-u',str(HERE/'estimation_checks.py')]
            with (HERE/'estimation_checks.log').open('a',encoding='utf-8') as log:
                child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=HERE.parent.parent,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                atomic_json(HERE/'state.json',dict(status='post_training_diagnostics',parent_pid=os.getpid(),
                    child_pid=child.pid,updated=stamp(),command=cmd))
                code=child.wait()
            if code:raise RuntimeError(f'post-training diagnostics failed:exit{code}; training results retained')
        atomic_json(HERE/'state.json',dict(status='completed',pid=os.getpid(),updated=stamp(),assistant_review='pending'))
    except Exception:
        atomic_json(HERE/'state.json',dict(status='failed',pid=os.getpid(),updated=stamp(),error=traceback.format_exc()));raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()
if __name__=='__main__':main()
