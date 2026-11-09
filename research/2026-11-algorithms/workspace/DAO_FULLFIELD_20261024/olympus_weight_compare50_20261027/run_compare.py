from group_loop import *
import os,json,ctypes,msvcrt,traceback
FULL=HERE.parent/'olympus_full500_20261027'
NROOT=HERE.parent/'olympus_secant_20261027'
KROOT=HERE.parent/'olympus_constrained_solver_20261026'
ROUTES=('equal_loop','equal_search','unequal_search')

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def state(stage,**kw):common.atomic(HERE/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))

def summary(rows):
    result=dict(completed=len(rows),total=50,complete=len(rows)==50,metrics={},paired_changes={},
        scope='Same old exploratory50, all continuations start from saved N; not a fresh validation set or a from-scratch pipeline comparison',
        truth_in_inference=False,new_acquisitions=0,training_updates=0)
    flags={}
    for route in ('network','D','K','N')+ROUTES:
        f=np.asarray([r['metrics'][route]['fields'] for r in rows],dtype=float)
        v=np.asarray([r['metrics'][route]['valid'] for r in rows],dtype=bool)
        passed=v&np.isfinite(f).all(1)&(f<.07).all(1);center=v&np.isfinite(f[:,0])&(f[:,0]<.07)
        flags[route]=(passed,center,v)
        result['metrics'][route]=dict(fullfield=int(passed.sum()),center=int(center.sum()),invalid=int((~v).sum()),
            center_mean=float(np.nanmean(f[:,0])),edge_mean=float(np.nanmean(f[:,1:])))
    for source,target in [('N',r) for r in ROUTES]+[('equal_loop','equal_search'),('equal_search','unequal_search')]:
        a,b=flags[source],flags[target]
        result['paired_changes'][source+'_to_'+target]=dict(gained=[rows[i]['case'] for i in np.where(~a[0]&b[0])[0]],
            lost=[rows[i]['case'] for i in np.where(a[0]&~b[0])[0]],center_gained=int((~a[1]&b[1]).sum()),
            center_lost=int((a[1]&~b[1]).sum()),new_invalid=int((a[2]&~b[2]).sum()),fixed_invalid=int((~a[2]&b[2]).sum()))
    result['costs']={route:dict(actual_camera_calls=sum(r['stages'][route]['details']['actual_camera_calls'] for r in rows),
        cache_hits=sum(r['stages'][route]['details']['cache_hits'] for r in rows),
        logical_proposal_trials=sum(r['stages'][route]['details']['proposal_trials'] for r in rows),
        actual_network_calls=sum(r['stages'][route]['details']['network_calls'] for r in rows),
        accepted_steps=sum(sum(s['accepted'] for s in r['stages'][route]['details']['steps']) for r in rows),
        seconds=sum(r['stages'][route]['details']['seconds'] for r in rows),
        model_errors=[r['case'] for r in rows if 'error' in r['stages'][route]['details']]) for route in ROUTES}
    result['incremental_active_seconds']=sum(r['active_seconds'] for r in rows)
    result['timing_scope']='Cache-assisted sequential paired comparison including new scoring; not cold route latency. Reused renderings and network forwards disclosed separately.'
    return result

def main():
    if (HERE/'result.json').exists():print((HERE/'result.json').read_text());return
    if (HERE/'STOP_REQUESTED.json').exists():state('paused');return
    selection=read(HERE/'selection.json');ids=selection['ids']
    old_selection=read(HERE.parent/'olympus_lowcost_refiner_20261026/selection.json')
    protocol=dict(ids=ids,paper='Downloads/manuscript (2)_subsystem.tex',paper_lines=[309,315,321,325],
        meaning='beta_g weights physical group correction proposals; not optical field/error weights',
        groups=7,coordinates_per_group=5,starting_point='Exact saved N command for every route',
        proposal='Frozen FP32 g(y)-g(F(x)), repartitioned into7 five-coordinate groups each cycle',
        observation='Same original noisy9-field packet; synthetic model calls are not new acquisitions',
        equal_loop='beta all1, unit common gain, reject non-improving move; no step search',
        equal_search='beta all1;16 common gains per cycle, zero implicit',
        unequal_search='Seven independent beta in[-2,2];common +/-1 followed by one +/-1 coordinate poll per group per cycle;zero implicit',
        max_cycles=4,search_trials_per_cycle=16,max_search_proposals_per_case=64,max_equal_loop_proposals=4,
        shared_constraints=dict(prior_radius=.2,step_max=.05,base_proposal_max=.025,center_limit='Frozen K terminal nominal proxy'),
        objective='Unchanged original per-pixel noise whitening plus same prior precision; no field reweighting',
        budget_note='Both search routes have the same4 cycle/64 trial caps and same network proposer. Plain loop is intentionally cheaper; early stops and constraint rejects do not trigger dummy work.',
        geometry='Common scalar feasibility clipping preserves relative beta; original geometry/path checks',
        score='Original centerABS and signed edgeDIFF; all9<.07 with original validity',
        full500_automatic=False,training_updates=0,truth_in_inference=False,automatic_deployment=False)
    common.atomic(HERE/'protocol.json',protocol)
    for name in ('cases','stages'): (HERE/name).mkdir(exist_ok=True)
    for route in ROUTES:(HERE/'stages'/route).mkdir(exist_ok=True)
    state('initializing',completed=len(list((HERE/'cases').glob('case_*.json'))),total=50)
    tick=common.clock();engine=GroupLoop();blocks=Blocks();metric=DiffMetrics();hidden=poses()
    common.atomic(HERE/'initialization.json',dict(seconds=common.clock()-tick,pid=os.getpid()))
    rows=[]
    for case in ids:
        output=HERE/'cases'/f'case_{case}.json'
        if output.exists():rows.append(read(output));continue
        if (HERE/'STOP_REQUESTED.json').exists():state('paused',completed=len(rows),total=50);return
        old=read(FULL/'cases'/f'case_{case}.json')
        kp=FULL/'stages/K'/f'case_{case}.json'
        if kp.exists():k=read(kp)['details']
        else:
            tag='development_k_adaptive8' if case in old_selection['development'] else 'paired32_k_adaptive8'
            k=read(KROOT/tag/f'case_{case}.json')
        prior=-np.asarray(old['commands']['network']);start=-np.asarray(old['commands']['N'])
        limit=k.get('final_center_limit',k.get('center_limit',.01))
        engine.begin_case(case,start);images,_=blocks.read('joint',case//25);stages={};tick=common.clock()
        for route in ROUTES:
            path=HERE/'stages'/route/f'case_{case}.json'
            if path.exists():stages[route]=read(path);continue
            if (HERE/'STOP_REQUESTED.json').exists():state('paused',completed=len(rows),total=50,case=case);return
            state('comparing',completed=len(rows),total=50,case=case,route=route)
            result=engine.predict(images[case%25],prior,start,limit,route)
            command=result.pop('command');stage=dict(command=command.tolist(),details=result)
            common.atomic(path,stage);stages[route]=stage
        # No hidden pose or performance score enters any of the three searches.
        scores=old['metrics'].copy();commands=old['commands'].copy()
        for route in ROUTES:
            command=stages[route]['command'];equal=next((r for r in scores if np.array_equal(command,commands[r])),None)
            if equal is not None:scores[route]=scores[equal].copy()
            else:
                value,valid=metric.score(hidden[case],np.asarray(command));f=value[0]
                scores[route]=dict(fields=[float(v) if np.isfinite(v) else None for v in f],valid=bool(valid[0] and np.isfinite(f).all()))
            commands[route]=command
        row=dict(case=case,metrics=scores,stages=stages,active_seconds=common.clock()-tick)
        common.atomic(output,row);rows.append(row);report=summary(rows);common.atomic(HERE/'partial_summary.json',report)
        eta=float(np.mean([r['active_seconds'] for r in rows][-10:]))*(50-len(rows))
        state('case_completed',completed=len(rows),total=50,last_case=case,estimated_remaining_seconds=eta)
        print(json.dumps(dict(case=case,completed=len(rows),fullfield={r:report['metrics'][r]['fullfield'] for r in ('N',)+ROUTES},
            seconds=row['active_seconds'],eta_seconds=eta)),flush=True)
    report=summary(rows);common.atomic(HERE/'result.json',report)
    state('computations_completed',completed=50,total=50,assistant_interpretation_pending=True,automatic_full500=False)
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
