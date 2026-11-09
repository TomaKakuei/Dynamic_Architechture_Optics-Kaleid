"""Fixed50 paired pilot: saved one-shot, loop1, loop2; no training."""
from loop import *
from optics import Optics
import os, traceback
TOTAL=50

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def state(stage,**kw):atomic(ROOT/'state.json',dict(stage=stage,pid=os.getpid(),updated=stamp(),**kw))
def stopped():return (ROOT/'STOP_REQUESTED.json').exists()

def aggregate(rows):
    report=dict(completed=len(rows),total=TOTAL,complete=len(rows)==TOTAL,metrics={},paired={},
        training_updates=0,new_acquisitions=0,truth_in_inference=False,max_loop_updates=2,
        seconds=sum(r['active_seconds'] for r in rows),
        camera_calls=sum(r['inference']['actual_camera_calls'] for r in rows),
        cache_hits=sum(r['inference']['cache_hits'] for r in rows),
        network_calls=sum(r['inference']['network_calls'] for r in rows),
        model_fallback_cases=[r['case'] for r in rows if 'error' in r['inference']],
        trial_model_errors=sum('error' in t for r in rows for s in r['inference']['steps'] for t in s.get('trials',[])),
        scope='Fixed50 from the previous600 selection set; single unchanged measured packet; not fresh independent validation',
        automatic_full3000=False,automatic_training=False)
    flags={}
    for arm in ('one_shot','loop1','loop2'):
        score=np.array([r['metrics'][arm]['score'] for r in rows]);valid=np.array([r['metrics'][arm]['valid'] for r in rows])
        passed=valid&(score>=.45);flags[arm]=passed
        report['metrics'][arm]=dict(passes=int(passed.sum()),rate=float(passed.mean()),invalid=int((~valid).sum()),q_mean=float(score.mean()),
            wrms_mean=float(np.mean([r['metrics'][arm]['wrms_mean'] for r in rows])))
    for left,right in (('one_shot','loop1'),('one_shot','loop2'),('loop1','loop2')):
        a,b=flags[left],flags[right]
        report['paired'][f'{left}_to_{right}']=dict(gained=[rows[i]['case'] for i in np.where(~a&b)[0]],lost=[rows[i]['case'] for i in np.where(a&~b)[0]])
    return report

def main():
    if (ROOT/'result.json').exists():print((ROOT/'result.json').read_text());return
    if stopped():state('stopped');return
    tick=clock();state('initializing')
    with np.load(INCUMBENT/'selected_validation3000_grid129.npz') as z:saved={k:z[k].copy() for k in z.files}
    with np.load(INCUMBENT/'selected_refined_scores.npz') as z:refined={k:z[k].copy() for k in z.files}
    with np.load(SOURCE/'training/joint_validation3000.npz') as z:old={k:z[k].copy() for k in z.files}
    with np.load(INCUMBENT/'selection.npz') as z:pool=z['valrows'].copy()
    selection=ROOT/'selection.json'
    if selection.exists():selected=read(selection)['rows']
    else:
        rng=np.random.default_rng(2026102850)
        selected=np.concatenate([rng.choice(pool[old['cohort'][pool]==c],n,replace=False) for c,n in enumerate((17,17,16))]).tolist()
        atomic(selection,dict(rows=selected,ids=old['indices'][selected].tolist(),selection='Fixed random17/17/16 per cohort from previous600; not selected by outcome'))
    atomic(ROOT/'protocol.json',dict(cases=TOTAL,selection=read(selection),checkpoint=str(INCUMBENT/'old_wfe_stable/candidate.pt'),
        arms=['one_shot_saved','loop1','loop2'],max_updates=2,proposal='x + gain*(g(y)-g(F(x))); same frozen g, original y fixed',
        gains=[1.,.5,-.5],max_step_in_normalized_coordinate=.05,max_departure_from_prior=.1,estimate_bounds=[-1.,1.],
        selection_objective='Mean same-packet squared residual / calibrated observed shot+read variance; no target MTF or hidden pose',
        noise_stop=1.05,accept='Best actually rendered candidate strictly lowers image merit; otherwise retain current estimate',
        geometry='Original known estimated-state/path checks; true endpoint/path scored only after inference',
        camera='Original fixed exposure/PRNU/DSNU, degree24 wave fit,2048 pupil/4096FFT; original4096/8192 bandwidth fallbacks',
        synthetic_input='Noise-free expected ADU approximation, no per-frame normalization or acquisition seed',
        score='Original fixed-image Q2000>=.45,PT removed/defocus retained,grid129 plus257 for near-threshold or65/129 disagreement',
        baseline='Saved original1684/3000 checkpoint scores; no baseline neural rerun',training_updates=0,new_acquisitions=0,
        truth_in_inference=False,automatic_full3000=False,automatic_training=False))
    images=np.load(SOURCE/'cache/images.npy',mmap_mode='r')
    engine=PredictorLoop();metrics={n:Optics(grid=n,device='cuda') for n in (65,129,257)}
    geometry=engine.geometry
    (ROOT/'cases').mkdir(exist_ok=True)
    atomic(ROOT/'initialization.json',dict(seconds=clock()-tick,pid=os.getpid()))
    @torch.no_grad()
    def score(initial,command):
        endpoint=initial+command;q=torch.tensor(endpoint[None],device='cuda',dtype=torch.float64)
        fine=metrics[129].metrics(q);coarse=metrics[65].metrics(q)
        v=float(fine['score'][0]);v65=float(coarse['score'][0]);valid=bool(fine['valid'][0] and coarse['valid'][0] and geometry.path(endpoint,initial))
        grid=129
        if abs(v-.45)<.02 or (v>=.45)!=(v65>=.45):
            dense=metrics[257].metrics(q);v=float(dense['score'][0]);valid=valid and bool(dense['valid'][0]);grid=257
        return dict(score=v,valid=valid,wrms_mean=float(fine['wrms'].mean()),grid=grid)
    records=[]
    for row in selected:
        case=int(old['indices'][row]);path=ROOT/'cases'/f'{case}.json'
        if path.exists():records.append(read(path));continue
        if stopped():state('stopped',completed=len(records),total=TOTAL);return
        start=clock();state('prediction_loop',completed=len(records),total=TOTAL,case=case)
        # Controller gets only observation and saved neural prior, never initial.
        prediction=engine.predict(np.asarray(images[case]),-saved['command'][row],case)
        scores={'one_shot':dict(score=float(refined['score'][row]),valid=bool(refined['valid'][row]),wrms_mean=float(saved['wrms'][row].mean()),grid='saved')}
        commands={'one_shot':saved['command'][row].copy()}
        for k,arm in enumerate(('loop1','loop2')):
            cmd=np.asarray(prediction['commands'][k]);same=next((a for a in commands if np.array_equal(commands[a],cmd)),None)
            scores[arm]=scores[same].copy() if same else score(old['initial'][row],cmd)
            commands[arm]=cmd
        record=dict(case=case,row=row,metrics=scores,inference=prediction,active_seconds=clock()-start)
        atomic(path,record);records.append(record);report=aggregate(records);atomic(ROOT/'partial_summary.json',report)
        recent=[r['active_seconds'] for r in records[-10:]];eta=float(np.mean(recent))*(TOTAL-len(records))
        state('case_complete',completed=len(records),total=TOTAL,last_case=case,estimated_remaining_seconds=eta)
        print(json.dumps(dict(completed=len(records),case=case,passes={a:report['metrics'][a]['passes'] for a in report['metrics']},seconds=record['active_seconds'],remaining_seconds=eta)),flush=True)
    report=aggregate(records);atomic(ROOT/'result.json',report)
    state('complete',completed=TOTAL,total=TOTAL,active_seconds=clock()-tick,assistant_interpretation='pending')
    print(json.dumps(report),flush=True)

if __name__=='__main__':
    awake()
    try:main()
    except Exception as error:state('failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:awake(False)
