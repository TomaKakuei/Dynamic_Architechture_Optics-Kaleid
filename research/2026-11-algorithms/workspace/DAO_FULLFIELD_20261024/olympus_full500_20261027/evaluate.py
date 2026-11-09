"""Frozen D -> K -> N on the original500; reuse completed cases and stage checkpoints."""
from pathlib import Path
import sys,os,json,ctypes,traceback,msvcrt,time
HERE=Path(__file__).resolve().parent
NROOT=HERE.parent/'olympus_secant_20261027'
KROOT=HERE.parent/'olympus_constrained_solver_20261026'
DROOT=HERE.parent/'olympus_lowcost_refiner_20261026'
sys.path.insert(0,str(NROOT))
from secant import SecantRefiner,Refiner,common,np,torch,Blocks,DiffMetrics,baseline,poses
from solver import ConstrainedRefiner
KTAG='evaluation500_k_adaptive8'
ROUTES=('network','D','K','N')


class LastImageCache:
    """Reuse one identical synthetic image between adjacent frozen stages."""
    def __init__(self,raw):
        self.raw=raw;self.last_q=None;self.last_output=None;self.calls=0;self.hits=0
    def __getattr__(self,name):return getattr(self.raw,name)
    def evaluate(self,q,derivative=False,callback=None):
        if derivative or callback is not None:return self.raw.evaluate(q,derivative=derivative,callback=callback)
        q=np.asarray(q)
        if self.last_q is not None and np.array_equal(q,self.last_q):
            self.hits+=1;return self.last_output
        self.calls+=1
        result=self.raw.evaluate(q)
        self.last_q=q.copy();self.last_output=result
        return result


def state(stage,**kw):
    common.atomic(HERE/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def summary(rows,final=False):
    out=dict(completed=len(rows),total=500,complete=final,route_metrics={},paired_changes={},
        reused_completed_cases=sum(r['runtime']['reused_archived_case'] for r in rows),
        newly_evaluated_cases=sum(not r['runtime']['reused_archived_case'] for r in rows),
        new_active_seconds=sum(r['runtime'].get('active_seconds',0) for r in rows),
        new_camera_evaluations=sum(r['runtime'].get('camera_evaluations',0) for r in rows),
        interstage_image_cache_hits=sum(r['runtime'].get('image_cache_hits',0) for r in rows),
        model_error_stages=[dict(case=r['case'],stages=r.get('model_error_stages',[])) for r in rows if r.get('model_error_stages')],
        criterion='Original center ABS + signed edge DIFF; all9<.07 waves and original validity',
        data_scope='Original500 validation; includes44 previously used development/comparison cases',
        timing_scope='New456 D/K/N model and offline scoring work; archived44 and saved neural inference excluded; not cold end-to-end latency',
        truth_in_inference=False,new_acquisitions=0,training_updates=0,automatic_deployment=False)
    flags={}
    for name in ROUTES:
        v=np.asarray([r['metrics'][name]['fields'] for r in rows],dtype=float)
        valid=np.asarray([r['metrics'][name]['valid'] for r in rows],dtype=bool)
        full=valid&np.isfinite(v).all(1)&(v<.07).all(1)
        center=valid&np.isfinite(v[:,0])&(v[:,0]<.07);flags[name]=(full,center,valid)
        out['route_metrics'][name]=dict(fullfield=int(full.sum()),center=int(center.sum()),invalid=int((~valid).sum()),
            fullfield_rate=float(full.mean()),center_mean=float(np.nanmean(v[:,0])),edge_mean=float(np.nanmean(v[:,1:])),
            finite_field_count=int(np.isfinite(v).sum()),per_field_passes=(valid[:,None]&np.isfinite(v)&(v<.07)).sum(0).tolist())
    for source in ('network','D','K'):
        a,b=flags[source],flags['N']
        out['paired_changes'][source+'_to_N']=dict(gained=[rows[i]['case'] for i in np.where(~a[0]&b[0])[0]],
            lost=[rows[i]['case'] for i in np.where(a[0]&~b[0])[0]],
            center_gained=int((~a[1]&b[1]).sum()),center_lost=int((a[1]&~b[1]).sum()),
            new_invalid=int((a[2]&~b[2]).sum()),fixed_invalid=int((~a[2]&b[2]).sum()))
    return out


def reuse_completed(old,slots):
    reused=[]
    for ntag,ktag,dtag in [('development_n_expand4','development_k_adaptive8','development_d8_center'),
                          ('paired32_n_expand4','paired32_k_adaptive8','confirmation_d8_center')]:
        for path in sorted((NROOT/ntag).glob('case_*.json')):
            n=read(path);case=n['case']
            if case not in slots:raise ValueError('Archived case outside original500')
            reused.append(case);output=HERE/'cases'/f'case_{case}.json'
            if output.exists():continue
            k=read(KROOT/ktag/f'case_{case}.json');d=read(DROOT/dtag/f'case_{case}.json')
            common.atomic(output,dict(case=case,commands=dict(network=old['command'][slots[case]].tolist(),D=d['command'],K=k['command'],N=n['command']),
                metrics=dict(network=n['network'],D=n['lowcost'],K=n['previous'],N=n['candidate']),
                provenance=dict(kind='reused_complete_N',source=str(path)),
                runtime=dict(reused_archived_case=True,active_seconds=0,camera_evaluations=0,image_cache_hits=0)))
    if len(reused)!=44 or len(set(reused))!=44:raise ValueError('Expected44 distinct completed N cases')
    return sorted(reused)


def stopped():return (HERE/'STOP_REQUESTED.json').exists()


def execute_stage(route,case,camera,invoke):
    path=HERE/'stages'/route/f'case_{case}.json'
    if path.exists():return read(path)
    start=common.clock();calls,hits=camera.calls,camera.hits
    result=invoke();command=result.pop('command')
    stage=dict(case=case,route=route,command=np.asarray(command).tolist(),details=result,
        active_seconds=common.clock()-start,camera_evaluations=camera.calls-calls,image_cache_hits=camera.hits-hits)
    common.atomic(path,stage)
    return stage


def format_score(values,valid):
    values=np.asarray(values,dtype=float)
    finite=np.isfinite(values)
    return dict(fields=[float(v) if np.isfinite(v) else None for v in values],
                valid=bool(valid and finite.all()),nonfinite_fields=int((~finite).sum()))


def main():
    if (HERE/'result.json').exists():print((HERE/'result.json').read_text());return
    old=baseline();ids=[int(i) for i in old['index']]
    if len(ids)!=500 or sorted(ids)!=list(range(7000,7500)):raise ValueError('Original500 ID contract mismatch')
    slots={case:i for i,case in enumerate(ids)}
    reused=reuse_completed(old,slots)
    rows=[read(HERE/'cases'/f'case_{case}.json') for case in ids if (HERE/'cases'/f'case_{case}.json').exists()]
    common.atomic(HERE/'protocol.json',dict(ids=ids,reused_ids=reused,missing_cases=456,
        scope='Frozen original500 validation, one existing noisy nine-field packet per case; no new neural inference or training',
        neural_predictions=str(NROOT.parent/'olympus_repair_4ep_20261026/evaluation500/predictions.npz'),
        D=dict(basis='hybrid8',steps=3,radius=.1,step_radius=.05,center_limit=.01),
        K=dict(basis='central8',h=.005,steps=3,adaptive_center=[.01,.02,.04],radius=.2),
        N=dict(secant_old_directions=8,new_central_directions=4,h=.005,steps=2,center_limit='K terminal',radius=.2),
        scoring='CenterABS + signed edgeDIFF, all9<.07 plus original endpoint/path/trace validity',
        labels_in_inference=False,renderer='2048pupil /4096FFT, full moving pupil response',
        engineering_only_changes='Shared immutable camera/calibration resources; one-entry exact-pose synthetic-image reuse; per-stage resume',
        automatic_followup=False,automatic_deployment=False))
    common.atomic(HERE/'partial_summary.json',summary(rows))
    state('initializing',completed=len(rows),reused_cases=len(reused),new_completed=len(rows)-len(reused),total=500)
    if stopped():state('paused',completed=len(rows),total=500);return
    t0=common.clock();d=Refiner('hybrid');d.camera=LastImageCache(d.camera);d.cache_dir=HERE/'responses_d';d.cache_dir.mkdir(exist_ok=True)
    k=ConstrainedRefiner(KTAG,'central',8,resources=d);n=SecantRefiner(resources=d)
    blocks=Blocks();metric=DiffMetrics();hidden_poses=poses()
    common.atomic(HERE/'initialization.json',dict(seconds=common.clock()-t0,shared_camera=True,pid=os.getpid()))
    for case in ids:
        if (HERE/'cases'/f'case_{case}.json').exists():continue
        if stopped():state('paused',completed=len(rows),total=500);return
        slot=slots[case];images,_=blocks.read('joint',case//25);image=images[case%25];prior=-old['command'][slot]
        progress=dict(case=case,completed=len(rows),reused_cases=44,new_completed=len(rows)-44,total=500)
        state('evaluating',phase='D',**progress)
        dr=execute_stage('D',case,d.camera,lambda:d.predict(image,prior,case,rank=8,steps=3,radius=.1,step_radius=.05,prior_strength=1.,center_limit=.01))
        if stopped():state('paused',phase='D_complete',**progress);return
        state('evaluating',phase='K',**progress)
        kr=execute_stage('K',case,d.camera,lambda:k.predict(image,prior,-np.asarray(dr['command']),case,cycles=3,radius=.2,center_limit=.01,adaptive=True))
        if stopped():state('paused',phase='K_complete',**progress);return
        physical=[s for s in kr['details']['steps'] if 'solver' in s]
        last=physical[-1]['cycle'] if physical and physical[-1]['accepted'] else None
        state('evaluating',phase='N',**progress)
        nr=execute_stage('N',case,d.camera,lambda:n.predict(image,prior,-np.asarray(kr['command']),case,KTAG,last,
            kr['details'].get('final_center_limit',kr['details'].get('center_limit',.01)),steps=2,extra=4))
        # Only after all candidate commands are fixed do labels enter scoring.
        state('scoring',**progress);score_start=common.clock()
        commands=dict(network=old['command'][slot].tolist(),D=dr['command'],K=kr['command'],N=nr['command'])
        scores=dict(network=format_score(old['mixed_wrms'][slot],old['valid'][slot]))
        for route in ('D','K','N'):
            equal=next((name for name in scores if np.array_equal(commands[route],commands[name])),None)
            if equal is not None:scores[route]=scores[equal].copy()
            else:
                values,valid=metric.score(hidden_poses[case],np.asarray(commands[route]))
                scores[route]=format_score(values[0],valid[0])
        stages=(dr,kr,nr)
        row=dict(case=case,commands=commands,metrics=scores,
            provenance=dict(kind='new_frozen_pipeline',stages=[str(HERE/'stages'/route/f'case_{case}.json') for route in ('D','K','N')]),
            model_error_stages=[s['route'] for s in stages if 'error' in s['details']],
            runtime=dict(reused_archived_case=False,active_seconds=sum(s['active_seconds'] for s in stages)+common.clock()-score_start,
                camera_evaluations=sum(s['camera_evaluations'] for s in stages),image_cache_hits=sum(s['image_cache_hits'] for s in stages)))
        common.atomic(HERE/'cases'/f'case_{case}.json',row);rows.append(row)
        complete=summary(rows);common.atomic(HERE/'partial_summary.json',complete)
        recent=[r['runtime']['active_seconds'] for r in rows if not r['runtime']['reused_archived_case']][-20:]
        eta=float(np.mean(recent))*(500-len(rows)) if len(recent)>=3 else None
        state('case_completed',completed=len(rows),reused_cases=44,new_completed=len(rows)-44,total=500,
              last_case=case,new_active_seconds=complete['new_active_seconds'],estimated_remaining_seconds=eta)
        print(json.dumps(dict(case=case,completed=len(rows),fullfield={route:bool(scores[route]['valid'] and all(v is not None and v<.07 for v in scores[route]['fields'])) for route in ROUTES},
            seconds=row['runtime']['active_seconds'],camera_calls=row['runtime']['camera_evaluations'],cache_hits=row['runtime']['image_cache_hits'],eta_seconds=eta)),flush=True)
    rows=sorted(rows,key=lambda r:r['case']);result=summary(rows,final=True)
    data=dict(index=np.asarray([r['case'] for r in rows]))
    for route in ROUTES:
        data['command_'+route]=np.asarray([r['commands'][route] for r in rows])
        data['fields_'+route]=np.asarray([r['metrics'][route]['fields'] for r in rows],dtype=float)
        data['valid_'+route]=np.asarray([r['metrics'][route]['valid'] for r in rows])
    np.savez_compressed(HERE/'predictions500.npz',**data)
    common.atomic(HERE/'result.json',result)
    lines=['Frozen Olympus original500 complete; generated results await assistant interpretation.',
           'Original centerABS + signed edgeDIFF, all9<.07 and original validity; one observed packet.',
           '44 completed cases reused;456 evaluated with unchanged frozen D/K/N rules.']
    for route in ROUTES:
        m=result['route_metrics'][route]
        lines.append(f"{route}: fullfield {m['fullfield']}/500; center {m['center']}/500; invalid {m['invalid']}; edge mean {m['edge_mean']:.9f}")
    lines.extend(['Full per-case data in predictions500.npz and cases/.',
        'Timing excludes archived44 and saved neural inference; not cold end-to-end latency.',
        'Validation includes previously used development/comparison cases, not a fresh independent test.',
        'No automatic next experiment, training or deployment.'])
    (HERE/'GENERATED_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state('computations_completed',completed=500,total=500,assistant_interpretation_pending=True,
          result='result.json',automatic_followup=False,automatic_deployment=False)


if __name__=='__main__':
    with (HERE/'run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        kernel=ctypes.windll.kernel32
        kernel.SetThreadExecutionState(0x80000001)
        try:main()
        except BaseException:
            state('failed',error=traceback.format_exc());raise
        finally:
            kernel.SetThreadExecutionState(0x80000000)
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
