"""One sequential resumable pipeline: camera context,3epoch residual training,200case score."""
from pathlib import Path
import sys,os,json,subprocess,traceback,msvcrt
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'model_residual'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT.parent))
from durable_io import atomic_json

def stamp():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def stopped():return (ROOT/'STOP_REQUESTED.json').exists() or (OUT/'STOP_REQUESTED.json').exists()
def state(stage,**kw):
    try:atomic_json(ROOT/'state.json',dict(stage=stage,pid=os.getpid(),updated=stamp(),**kw))
    except PermissionError as exc:print(f'Telemetry deferred: {exc!r}',flush=True)

def child(script,args,stage,log_prefix):
    with (OUT/(log_prefix+'stdout.log')).open('a',encoding='utf-8') as so, (OUT/(log_prefix+'stderr.log')).open('a',encoding='utf-8') as se:
        p=subprocess.Popen([sys.executable,'-u',str(ROOT/script),*args],cwd=str(ROOT),
            stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        state(stage,child_pid=p.pid,script=script)
        print(json.dumps(dict(stage=stage,child_pid=p.pid,started=stamp())),flush=True)
        return p.wait()

def summarize():
    import numpy as np
    arm=read(OUT/'result.json')
    cache=read(OUT/'cache_complete.json')
    def paired_saved(path):
        with np.load(path) as z:
            old=z['valid'] & (z['score']>=.45);ids=z['indices'].copy()
        with np.load(OUT/'candidate200.npz') as z:
            assert np.array_equal(ids,z['indices'])
            new=z['valid'] & (z['score']>=.45)
        return dict(control_passes=int(old.sum()),candidate_passes=int(new.sum()),
            gained=ids[~old&new].tolist(),lost=ids[old&~new].tolist(),net=int(new.sum()-old.sum()))
    result=dict(complete=True,assistant_interpretation='pending',updated=stamp(),
        full27000_prior=arm['bounded_control'],candidate=arm['candidate'],
        direct_vs_full27000=arm['direct_vs_control'],
        historical_small_data_control=paired_saved(ROOT/'historical_small_data_control200.npz'),
        historical_old_prior_residual=paired_saved(ROOT/'historical_old_prior_residual200.npz'),
        training_samples=27000,epochs=3,optimizer_updates=2532,
        training_active_seconds=arm['training_active_seconds'],evaluation_active_seconds=arm['evaluation_active_seconds'],
        camera_context=cache,scope='Same reused200cases,one acquired packet,unchanged old loss; frozen full27000model plus learned residual head',
        automatic_full3000=False,automatic_deployment=False,automatic_further_training=False,
        limitations=read(ROOT/'protocol.json')['attribution_limits'])
    atomic_json(ROOT/'result.json',result)
    a=result['candidate'];b=result['full27000_prior'];p=result['direct_vs_full27000']
    lines=['KLA full27000 model + full27000-trained residual head,3epochs.',
        'Generated results; assistant interpretation pending. Same reused200selection cases.',
        f"Frozen full-data prior: {b['passes']}/200; mean WRMS {b['wrms_mean']:.8f}; invalid {b['invalid']}.",
        f"Plus residual correction: {a['passes']}/200; mean WRMS {a['wrms_mean']:.8f}; invalid {a['invalid']}.",
        f"Paired gain/loss: +{p['gained']}/-{p['lost']}.",
        'One measured packet, extra native model render, no physical feedback; no automatic deployment.']
    (ROOT/'GENERATED_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    if (ROOT/'result.json').exists():
        state('complete',assistant_interpretation=read(ROOT/'result.json').get('assistant_interpretation','pending'));return
    if stopped():state('stopped_before_start');return
    if not (OUT/'cache_complete.json').exists():
        rc=child('prepare_camera_cache.py',[],'preparing_model_context','cache.')
        if rc or not (OUT/'cache_complete.json').exists():
            state('stopped' if stopped() else 'cache_failed',child_exit_code=rc);return
    if stopped():state('stopped_after_cache');return
    if not (OUT/'result.json').exists():
        rc=child('train_arm.py',['model_residual'],'training_then_evaluating200','')
        if rc or not (OUT/'result.json').exists():
            state('stopped' if stopped() else 'training_or_evaluation_failed',child_exit_code=rc);return
    summarize()
    state('complete',assistant_interpretation='pending',automatic_followup=False)

if __name__=='__main__':
    with (ROOT/'suite.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:main()
        except Exception as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise
