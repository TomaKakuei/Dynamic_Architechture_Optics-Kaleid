from wide_search import *
import os,json,ctypes,msvcrt,traceback
FULL=HERE.parent/'olympus_full500_20261027'
OLDLINE=HERE.parent/'olympus_local_ablation150_20261027'
ROUTES=('N','previous_line','search64')

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def state(stage,**kw):common.atomic(HERE/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))
def stopped():return (HERE/'STOP_REQUESTED.json').exists()
def kdetails(case,selection):
    path=FULL/'stages/K'/f'case_{case}.json'
    if path.exists():return read(path)['details'],'evaluation500_k_adaptive8'
    tag='development_k_adaptive8' if case in selection['development'] else 'paired32_k_adaptive8'
    return read(KROOT/tag/f'case_{case}.json'),tag

def summary(rows):
    result=dict(completed=len(rows),total=150,complete=len(rows)==150,new_configurations=1,
        max_N_updates=2,max_model_evaluations_per_loop=64,metrics={},paired_changes={},
        active_search_cases=sum(r['active_search'] for r in rows),reused_inactive_cases=sum(not r['active_search'] for r in rows),
        total_active_seconds=sum(r['active_seconds'] for r in rows),truth_in_inference=False,new_acquisitions=0,training_updates=0,
        scope='Same150 reused cases;one new configuration from original K,original N/previous_line are saved references only',
        timing_scope='New joint-search/scoring,cache-assisted;NN/D/K/calibration/reference runs excluded,not cold complete latency')
    flags={}
    for route in ROUTES:
        f=np.asarray([r['metrics'][route]['fields'] for r in rows],dtype=float);v=np.asarray([r['metrics'][route]['valid'] for r in rows],dtype=bool)
        full=v&np.isfinite(f).all(1)&(f<.07).all(1);center=v&np.isfinite(f[:,0])&(f[:,0]<.07);flags[route]=(full,center,v)
        result['metrics'][route]=dict(fullfield=int(full.sum()),fullfield_rate=float(full.mean()),center=int(center.sum()),invalid=int((~v).sum()),
            center_mean=float(np.nanmean(f[:,0])),edge_mean=float(np.nanmean(f[:,1:])))
    for source in ('N','previous_line'):
        a,b=flags[source],flags['search64']
        result['paired_changes'][source+'_to_search64']=dict(gained=[rows[i]['case'] for i in np.where(~a[0]&b[0])[0]],
            lost=[rows[i]['case'] for i in np.where(a[0]&~b[0])[0]],center_gained=int((~a[1]&b[1]).sum()),center_lost=int((a[1]&~b[1]).sum()),
            new_invalid=int((a[2]&~b[2]).sum()),fixed_invalid=int((~a[2]&b[2]).sum()))
    d=[r['stage']['details'] for r in rows]
    result['costs']=dict(actual_camera_calls=sum(x['actual_camera_calls'] for x in d),cache_hits=sum(x['cache_hits'] for x in d),
        objective_requests=sum(x['objective_requests'] for x in d),accepted_updates=sum(sum(s['accepted'] for s in x['steps']) for x in d),
        search64_selected_updates=sum(sum(s.get('selected_source')=='joint_COBYQA' for s in x['steps']) for x in d),
        max_observed_outer_updates=max((sum(s['accepted'] for s in x['steps']) for x in d),default=0),
        max_observed_evaluations_per_loop=max((s['objective_requests'] for x in d for s in x['steps']),default=0),
        model_errors=[r['case'] for r in rows if 'error' in r['stage']['details']],
        budget_limited_searches=sum(sum(bool(s.get('search') and s['search'].get('budget_limited')) for s in x['steps']) for x in d))
    return result

def main():
    if (HERE/'result.json').exists():print((HERE/'result.json').read_text());return
    if stopped():state('paused');return
    ids=read(HERE/'selection.json')['ids'];oldselection=read(HERE.parent/'olympus_lowcost_refiner_20261026/selection.json')
    old={i:read(FULL/'cases'/f'case_{i}.json') for i in ids};ks={i:kdetails(i,oldselection) for i in ids}
    active=[]
    for i,(k,tag) in ks.items():
        physical=[s for s in k['steps'] if 'solver' in s]
        if physical and physical[-1]['accepted']:active.append(i)
    common.atomic(HERE/'protocol.json',dict(ids=ids,new_configurations=1,active_cases=active,max_N_updates=2,
        search_budget='64 objective requests per outer loop,128 per active case maximum,including original N seeding',
        search='Original N constrained secant seed then12D nonlinear COBYQA;virtual probes never change outer x until final selection',
        original_seed_gains=[1.,.5,.25],variable_scale=.05,initial_tr_radius=.25,final_tr_radius=.01,
        cobyqa_maxiter=1000,cobyqa_feasibility_tol=1e-8,
        unchanged=dict(rank=12,prior_radius=.2,step_limit=.05,center_limit='Original K terminal nominal proxy',
            observation='One original noisy9-field packet',objective='Original sigma whitening plus original prior precision',
            accept='Original actual-merit decrease and agreement>=.1;accepted point satisfies original constraints'),
        probe_note='Virtual COBYQA probes may violate prior/center trust constraints while fitting its search model;only physically safe poses are rendered,and only trust-feasible points may be accepted',
        start='Same saved K endpoint;not a continuation after original N2',
        previous_results='N112/150 and oldline113/150 reused as references;not rerun as additional configurations',
        scoring='Original centerABS+signed edgeDIFF all9<.07 plus original endpoint/path validity',
        no_new_FD=True,no_neural_loop=True,truth_in_inference=False,new_acquisitions=0,training_updates=0,
        automatic_full500=False,automatic_deployment=False))
    for name in ('cases','stages'):(HERE/name).mkdir(exist_ok=True)
    state('initializing',completed=0,total=150,active_cases=len(active))
    tick=common.clock();engine=WideSearch();blocks=Blocks();metric=DiffMetrics();hidden=poses()
    common.atomic(HERE/'initialization.json',dict(seconds=common.clock()-tick,pid=os.getpid()))
    rows=[]
    for case in ids:
        path=HERE/'cases'/f'case_{case}.json'
        if path.exists():rows.append(read(path));continue
        if stopped():state('paused',completed=len(rows),total=150);return
        tick=common.clock();base=old[case];stagepath=HERE/'stages'/f'case_{case}.json'
        if case not in active:
            stage=dict(command=base['commands']['N'],details=dict(steps=[],seconds=0.,actual_camera_calls=0,cache_hits=0,objective_requests=0,
                stop_reason='original_no_active_K_response_reused',max_updates=2))
        elif stagepath.exists():stage=read(stagepath)
        else:
            k,tag=ks[case];physical=[s for s in k['steps'] if 'solver' in s];last=physical[-1]['cycle']
            state('preparing_N_response',completed=len(rows),total=150,case=case)
            images,_=blocks.read('joint',case//25);engine.begin_case(case)
            prepared=engine.prepare(images[case%25],-np.asarray(base['commands']['network']),-np.asarray(base['commands']['K']),
                case,tag,last,k.get('final_center_limit',k.get('center_limit',.01)))
            def progress(loop,evaluations,cost):
                state('joint_search64',completed=len(rows),total=150,case=case,loop=loop,max_loops=2,
                    loop_evaluations=evaluations,max_loop_evaluations=64,best_image_merit=cost)
            result=engine.predict(prepared,progress=progress);command=result.pop('command')
            stage=dict(command=command.tolist(),details=result);common.atomic(stagepath,stage)
        # The candidate is fixed before loading scoring references or hidden poses.
        previous=read(OLDLINE/'cases'/f'case_{case}.json')
        scores=dict(N=base['metrics']['N'].copy(),previous_line=previous['metrics']['N_line'].copy())
        references=dict(N=base['commands']['N'],previous_line=previous['commands']['N_line'])
        equal=next((r for r,q in references.items() if np.array_equal(q,stage['command'])),None)
        if equal is not None:scores['search64']=scores[equal].copy()
        else:
            f,v=metric.score(hidden[case],np.asarray(stage['command']));values=f[0]
            scores['search64']=dict(fields=[float(x) if np.isfinite(x) else None for x in values],valid=bool(v[0] and np.isfinite(values).all()))
        row=dict(case=case,command=stage['command'],metrics=scores,stage=stage,active_search=case in active,active_seconds=common.clock()-tick)
        common.atomic(path,row);rows.append(row);report=summary(rows);common.atomic(HERE/'partial_summary.json',report)
        times=[r['active_seconds'] for r in rows if r['active_search']][-10:]
        remaining=len(active)-sum(r['active_search'] for r in rows);eta=float(np.mean(times))*remaining if times else None
        state('case_completed',completed=len(rows),total=150,last_case=case,remaining_active=remaining,estimated_remaining_seconds=eta)
        print(json.dumps(dict(case=case,completed=len(rows),fullfield={r:report['metrics'][r]['fullfield'] for r in ROUTES},
            active=case in active,seconds=row['active_seconds'],objective_requests=stage['details']['objective_requests'],eta_seconds=eta)),flush=True)
    report=summary(rows);common.atomic(HERE/'result.json',report)
    np.savez_compressed(HERE/'predictions150.npz',index=np.asarray([r['case'] for r in rows]),command=np.asarray([r['command'] for r in rows]),
        fields=np.asarray([r['metrics']['search64']['fields'] for r in rows],dtype=float),valid=np.asarray([r['metrics']['search64']['valid'] for r in rows]))
    m=report['metrics']['search64']
    (HERE/'GENERATED_REPORT.txt').write_text(f"Single search64 configuration complete;awaits assistant interpretation.\nN loops<=2,objective queries<=64perloop.\nFullfield{m['fullfield']}/150,center{m['center']},invalid{m['invalid']}.\n",encoding='utf-8')
    state('computations_completed',completed=150,total=150,assistant_interpretation_pending=True,automatic_full500=False)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with (HERE/'run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:main()
        except BaseException:
            state('failed',error=traceback.format_exc());raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
