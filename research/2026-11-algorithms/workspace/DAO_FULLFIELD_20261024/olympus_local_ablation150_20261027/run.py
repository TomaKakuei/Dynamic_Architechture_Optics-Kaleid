from search import *
import json,os,ctypes,msvcrt,traceback
FULL=HERE.parent/'olympus_full500_20261027'
ARMS=('N','N_line','N_pattern','N_line_pattern')

def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def state(stage,**kw):common.atomic(HERE/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))
def stopped():return (HERE/'STOP_REQUESTED.json').exists()
def kdetails(case,selection):
    path=FULL/'stages/K'/f'case_{case}.json'
    if path.exists():return read(path)['details'],'evaluation500_k_adaptive8'
    tag='development_k_adaptive8' if case in selection['development'] else 'paired32_k_adaptive8'
    return read(KROOT/tag/f'case_{case}.json'),tag

def summary(rows):
    result=dict(completed=len(rows),total=150,complete=len(rows)==150,max_N_updates=2,metrics={},paired_changes={},costs={},
        active_search_cases=sum(r['active_search'] for r in rows),reused_inactive_cases=sum(not r['active_search'] for r in rows),
        total_active_seconds=sum(r['active_seconds'] for r in rows),truth_in_inference=False,new_acquisitions=0,training_updates=0,
        scope='Same150 reused-validation cases as previous N budget experiment;2x2 local-method ablation inside N2 from common K endpoint',
        timing_scope='Sequential cache-assisted new search/scoring only; old N baseline/calibration/NN/D/K reused. Logical requests and cache hits reported; not cold latency.')
    flags={}
    for arm in ARMS:
        f=np.asarray([r['metrics'][arm]['fields'] for r in rows],dtype=float)
        v=np.asarray([r['metrics'][arm]['valid'] for r in rows],dtype=bool)
        full=v&np.isfinite(f).all(1)&(f<.07).all(1);center=v&np.isfinite(f[:,0])&(f[:,0]<.07);flags[arm]=(full,center,v)
        result['metrics'][arm]=dict(fullfield=int(full.sum()),fullfield_rate=float(full.mean()),center=int(center.sum()),invalid=int((~v).sum()),
            center_mean=float(np.nanmean(f[:,0])),edge_mean=float(np.nanmean(f[:,1:])))
        if arm!='N':
            d=[r['stages'][arm]['details'] for r in rows]
            result['costs'][arm]=dict(actual_camera_calls=sum(x['actual_camera_calls'] for x in d),
                cache_hits=sum(x['cache_hits'] for x in d),proposal_attempts=sum(x['proposal_attempts'] for x in d),
                seconds=sum(x['seconds'] for x in d),accepted_updates=sum(sum(s['accepted'] for s in x['steps']) for x in d),
                model_errors=[r['case'] for r in rows if 'error' in r['stages'][arm]['details']],
                selected_direction_updates=sum(sum(s.get('selected_source')=='direction_poll' for s in x['steps']) for x in d))
    pairs=[('N',a) for a in ARMS[1:]]+[('N_line','N_line_pattern'),('N_pattern','N_line_pattern')]
    for source,target in pairs:
        a,b=flags[source],flags[target]
        result['paired_changes'][source+'_to_'+target]=dict(gained=[rows[i]['case'] for i in np.where(~a[0]&b[0])[0]],
            lost=[rows[i]['case'] for i in np.where(a[0]&~b[0])[0]],center_gained=int((~a[1]&b[1]).sum()),
            center_lost=int((a[1]&~b[1]).sum()),new_invalid=int((a[2]&~b[2]).sum()),fixed_invalid=int((~a[2]&b[2]).sum()))
    result['fullfield_factorial_interaction_count']=(result['metrics']['N_line_pattern']['fullfield']-result['metrics']['N_line']['fullfield']
        -result['metrics']['N_pattern']['fullfield']+result['metrics']['N']['fullfield'])
    return result

def main():
    if (HERE/'result.json').exists():print((HERE/'result.json').read_text());return
    if stopped():state('paused');return
    ids=read(HERE/'selection.json')['ids'];oldselection=read(HERE.parent/'olympus_lowcost_refiner_20261026/selection.json')
    records={i:read(FULL/'cases'/f'case_{i}.json') for i in ids}
    ks={i:kdetails(i,oldselection) for i in ids};active=[]
    for i,(k,tag) in ks.items():
        physical=[s for s in k['steps'] if 'solver' in s]
        if physical and physical[-1]['accepted']:active.append(i)
    common.atomic(HERE/'protocol.json',dict(ids=ids,arms=list(ARMS),active_cases=active,max_N_updates=2,
        design='2x2 factorial: enhanced scalar line search on/off; bounded coordinate-direction polls on/off',
        starting_point='Same archived K endpoint; replace search INSIDE N first/second updates, never append updates after N2',
        baseline='Original N2 saved commands/scores,112/150',
        N='Original constrained secant direction; gains1/.5/.25,first acceptable',
        N_line='Same direction; evaluate1/.5/.25/.125/.0625 plus at most1 bounded quadratic interpolation;best acceptable',
        N_pattern='Original N gain search plus signed polls on4 model-gradient-ranked axes in original12D response;best acceptable',
        N_line_pattern='Both candidate families at each same loop starting point;accept at mostone update',
        pattern='Rank abs(g)/sqrt(diag(Q));two signs peraxis,physical coordinate radius.025,common shortening to original constraints',
        max_model_proposals_per_update=dict(N=3,N_line=6,N_pattern=11,N_line_pattern=14),
        unchanged=dict(response_rank=12,prior_radius=.2,step_limit=.05,center_limit='Original K terminal nominal proxy',
            prior_precision='inv(cov)/128^2',measurement_weights='Original sigma_from_adu whitening',
            accept='Actual image/prior merit decreases and actual/predicted ratio>=.1',eligibility='Original last K response accepted'),
        score='CenterABS+signed edgeDIFF all9<.07 with original endpoint/path validity',
        budget_note='Fixed outer2 updates; candidate evaluation costs intentionally differ by method. Report actual calls,do not label equal compute.',
        no_neural_loop=True,no_group_gain_change=True,no_new_FD=True,truth_in_inference=False,new_acquisitions=0,training_updates=0,
        automatic_full500=False,automatic_deployment=False))
    for name in ('cases','stages'):(HERE/name).mkdir(exist_ok=True)
    for arm in ARMS[1:]:(HERE/'stages'/arm).mkdir(exist_ok=True)
    state('initializing',completed=0,total=150,active_cases=len(active))
    tick=common.clock();engine=SearchAblation();blocks=Blocks();metric=DiffMetrics();hidden=poses()
    common.atomic(HERE/'initialization.json',dict(seconds=common.clock()-tick,pid=os.getpid()))
    rows=[]
    for case in ids:
        path=HERE/'cases'/f'case_{case}.json'
        if path.exists():rows.append(read(path));continue
        if stopped():state('paused',completed=len(rows),total=150);return
        tick=common.clock();old=records[case];stages={}
        if case not in active:
            for arm in ARMS[1:]:stages[arm]=dict(command=old['commands']['N'],details=dict(steps=[],seconds=0.,actual_camera_calls=0,
                cache_hits=0,proposal_attempts=0,stop_reason='original_no_active_K_response_reused',max_updates=2))
        else:
            k,tag=ks[case];physical=[s for s in k['steps'] if 'solver' in s];last=physical[-1]['cycle']
            state('preparing_shared_N_response',completed=len(rows),total=150,case=case)
            images,_=blocks.read('joint',case//25);engine.begin_case(case)
            prepared=engine.prepare(images[case%25],-np.asarray(old['commands']['network']),-np.asarray(old['commands']['K']),
                case,tag,last,k.get('final_center_limit',k.get('center_limit',.01)))
            for arm in ARMS[1:]:
                stagepath=HERE/'stages'/arm/f'case_{case}.json'
                if stagepath.exists():stages[arm]=read(stagepath);continue
                if stopped():state('paused',completed=len(rows),total=150,case=case,arm=arm);return
                state('local_search_ablation',completed=len(rows),total=150,case=case,arm=arm,max_N_updates=2)
                result=engine.predict_arm(prepared,arm);command=result.pop('command')
                stage=dict(command=command.tolist(),details=result);common.atomic(stagepath,stage);stages[arm]=stage
        # Infer all candidate endpoints first; only then access hidden-pose scoring.
        scores={'N':old['metrics']['N'].copy()};commands={'N':old['commands']['N']}
        for arm in ARMS[1:]:
            command=stages[arm]['command'];equal=next((a for a in scores if np.array_equal(command,commands[a])),None)
            if equal is not None:scores[arm]=scores[equal].copy()
            else:
                f,v=metric.score(hidden[case],np.asarray(command));values=f[0]
                scores[arm]=dict(fields=[float(x) if np.isfinite(x) else None for x in values],valid=bool(v[0] and np.isfinite(values).all()))
            commands[arm]=command
        row=dict(case=case,commands=commands,metrics=scores,stages=stages,active_search=case in active,active_seconds=common.clock()-tick)
        common.atomic(path,row);rows.append(row);report=summary(rows);common.atomic(HERE/'partial_summary.json',report)
        times=[r['active_seconds'] for r in rows if r['active_search']][-10:]
        remaining=len(active)-sum(r['active_search'] for r in rows);eta=float(np.mean(times))*remaining if times else None
        state('case_completed',completed=len(rows),total=150,last_case=case,remaining_active=remaining,estimated_remaining_seconds=eta)
        print(json.dumps(dict(case=case,completed=len(rows),fullfield={a:report['metrics'][a]['fullfield'] for a in ARMS},
            active=case in active,seconds=row['active_seconds'],eta_seconds=eta)),flush=True)
    report=summary(rows);common.atomic(HERE/'result.json',report)
    data=dict(index=np.asarray([r['case'] for r in rows]))
    for arm in ARMS:
        data['command_'+arm]=np.asarray([r['commands'][arm] for r in rows]);data['fields_'+arm]=np.asarray([r['metrics'][arm]['fields'] for r in rows],dtype=float)
        data['valid_'+arm]=np.asarray([r['metrics'][arm]['valid'] for r in rows])
    np.savez_compressed(HERE/'predictions150.npz',**data)
    lines=['Fixed2-update original N local search factorial ablation on150; generated results await interpretation.']
    for arm in ARMS:
        m=report['metrics'][arm];lines.append(f"{arm}:fullfield{m['fullfield']}/150;center{m['center']};invalid{m['invalid']};center_mean{m['center_mean']:.9f};edge_mean{m['edge_mean']:.9f}")
    (HERE/'GENERATED_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
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
