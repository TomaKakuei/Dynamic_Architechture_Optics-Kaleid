"""Original N, one shared trajectory, four iteration budgets on fixed150."""
from pathlib import Path
import json,os,ctypes,msvcrt,traceback
from original_n_instrumented import SecantRefiner,common,np,torch,Blocks,DiffMetrics,poses
HERE=Path(__file__).resolve().parent
FULL=HERE.parent/'olympus_full500_20261027'
NROOT=HERE.parent/'olympus_secant_20261027'
KROOT=HERE.parent/'olympus_constrained_solver_20261026'
BUDGETS=(2,4,8,16)


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def state(stage,**kw):common.atomic(HERE/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))
def stopped():return (HERE/'STOP_REQUESTED.json').exists()
class PauseRequested(Exception):pass
class BaselineAlignmentError(Exception):pass


class CameraCache:
    """Reuse available exact-pose images; do not change original N arithmetic."""
    def __init__(self,raw):self.raw=raw;self.calls=0;self.hits=0;self.cached=[]
    def __getattr__(self,name):return getattr(self.raw,name)
    def begin_case(self,case):
        self.cached=[]
        path=HERE.parent/'olympus_weight_compare50_20261027/initial_images'/f'{case}.npz'
        if path.exists():
            with np.load(path) as z:self.cached.append((z['estimate'].copy(),z['image'].copy()))
    def evaluate(self,q,derivative=False,callback=None):
        if derivative or callback is not None:return self.raw.evaluate(q,derivative=derivative,callback=callback)
        q=np.asarray(q)
        for old,image in self.cached:
            if np.array_equal(old,q):
                self.hits+=1;return torch.as_tensor(image,device='cuda'),None
        self.calls+=1;result=self.raw.evaluate(q)
        self.cached.append((q.copy(),result[0].cpu().numpy()))
        return result


def old_details(case,route,selection):
    path=FULL/'stages'/route/f'case_{case}.json'
    if path.exists():return read(path)['details']
    split='development' if case in selection['development'] else 'paired32'
    root,suffix=(NROOT,'_n_expand4') if route=='N' else (KROOT,'_k_adaptive8')
    return read(root/(split+suffix)/f'case_{case}.json')


def eligible(n):
    steps=n.get('steps',[])
    return len(steps)==2 and bool(steps[-1]['accepted'])


def score_dict(fields,valid):
    fields=np.asarray(fields,dtype=float)
    return dict(fields=[float(x) if np.isfinite(x) else None for x in fields],
                valid=bool(valid and np.isfinite(fields).all()))


def summary(rows):
    out=dict(completed=len(rows),total=150,complete=len(rows)==150,metrics={},paired_changes={},costs={},
        new_trajectory_cases=sum(r['continued'] for r in rows),
        reused_early_stop_cases=sum(not r['continued'] for r in rows),
        original_method=True,budgets=list(BUDGETS),truth_in_inference=False,new_acquisitions=0,training_updates=0,
        model_errors=[r['case'] for r in rows if r.get('model_error')],
        scope='Fixed-seed150 from original500; reused validation, same cases for all budgets',
        timing_scope='Shared-prefix incremental work. Original2step scores and D/K/neural work reused. Prefix-state reconstruction is reported separately; not cold pipeline latency.')
    flags={}
    for budget in BUDGETS:
        key=str(budget);v=np.asarray([r['metrics'][key]['fields'] for r in rows],dtype=float)
        valid=np.asarray([r['metrics'][key]['valid'] for r in rows],dtype=bool)
        full=valid&np.isfinite(v).all(1)&(v<.07).all(1)
        center=valid&np.isfinite(v[:,0])&(v[:,0]<.07);flags[key]=(full,center,valid)
        out['metrics'][key]=dict(fullfield=int(full.sum()),fullfield_rate=float(full.mean()),center=int(center.sum()),
            invalid=int((~valid).sum()),center_mean=float(np.nanmean(v[:,0])),edge_mean=float(np.nanmean(v[:,1:])))
        out['costs'][key]=dict(
            extra_loop_model_requests=sum(r['prefix'][key]['extra_model_requests'] for r in rows),
            extra_new_renders=sum(r['prefix'][key]['extra_new_renders'] for r in rows),
            extra_loop_seconds=sum(r['prefix'][key]['extra_loop_seconds'] for r in rows),
            accepted_extra_steps=sum(r['prefix'][key]['accepted_extra_steps'] for r in rows),
            actual_iteration_counts=[r['prefix'][key]['actual_iterations'] for r in rows])
    for source,target in [('2',str(b)) for b in BUDGETS[1:]]+[('4','8'),('8','16')]:
        a,b=flags[source],flags[target]
        out['paired_changes'][source+'_to_'+target]=dict(
            gained=[rows[i]['case'] for i in np.where(~a[0]&b[0])[0]],
            lost=[rows[i]['case'] for i in np.where(a[0]&~b[0])[0]],
            center_gained=int((~a[1]&b[1]).sum()),center_lost=int((a[1]&~b[1]).sum()),
            new_invalid=int((a[2]&~b[2]).sum()),fixed_invalid=int((~a[2]&b[2]).sum()))
    out['state_reconstruction_seconds']=sum(r['reconstruction']['seconds'] for r in rows)
    out['state_reconstruction_new_renders']=sum(r['reconstruction']['new_renders'] for r in rows)
    out['new_total_active_seconds']=sum(r['active_seconds'] for r in rows)
    return out


def main():
    if (HERE/'result.json').exists():print((HERE/'result.json').read_text());return
    if stopped():state('paused');return
    selection=read(HERE/'selection.json');ids=selection['ids']
    archived_selection=read(HERE.parent/'olympus_lowcost_refiner_20261026/selection.json')
    records={i:read(FULL/'cases'/f'case_{i}.json') for i in ids}
    nrecords={i:old_details(i,'N',archived_selection) for i in ids}
    active_ids=[i for i in ids if eligible(nrecords[i])]
    common.atomic(HERE/'protocol.json',dict(ids=ids,budgets=list(BUDGETS),seed=selection['seed'],
        changed_parameter='N maximum iteration count only:2,4,8,16',
        unchanged=dict(search_rank=12,old_directions=8,extra_directions=4,central_h=.005,
            gains=[1.,.5,.25],prior_radius=.2,step_limit=.05,center_limit='Frozen K terminal',
            acceptance='Actual merit decreases and agreement>=.1',early_stop='Original infeasible/no-predicted-gain/rejected-step stop',
            observation_weights='Original sigma_from_adu whitening',prior_precision='Original inv(cov)/128^2'),
        shared_prefix=True,active_continuation_ids=active_ids,
        original_N_reference='Saved original2step commands/scores, no rescoring',
        reconstruction='Original final secant Jacobian was not saved. Active cases reconstruct first2 steps once from cached K response/N initial extra response; existing exact-pose terminal images are reused when available. This state-building work is shared by all larger budgets.',
        no_continuation_rule='If original N stopped before its2step cap, its original stop remains valid at every larger cap; reuse the exact saved endpoint without any rendering',
        scoring='Original centerABS + signed edgeDIFF all9<.07 plus original endpoint/path validity',
        scope='Uniform fixed-seed150 from original500; reused exploratory validation',
        automatic_full500=False,automatic_deployment=False,truth_in_inference=False,new_acquisitions=0,training_updates=0))
    for directory in ('cases','trajectories'):(HERE/directory).mkdir(exist_ok=True)
    state('initializing',completed=0,total=150,active_continuations=len(active_ids))
    t=common.clock();refiner=SecantRefiner();camera=CameraCache(refiner.camera);refiner.camera=camera
    blocks=Blocks();metric=DiffMetrics();hidden=poses()
    common.atomic(HERE/'initialization.json',dict(seconds=common.clock()-t,pid=os.getpid()))
    rows=[]
    for case in ids:
        path=HERE/'cases'/f'case_{case}.json'
        if path.exists():rows.append(read(path));continue
        if stopped():state('paused',completed=len(rows),total=150);return
        old=records[case];n=nrecords[case];tick=common.clock();commands={};prefix={};error=None
        reconstruction=dict(seconds=0.,new_renders=0)
        if case not in active_ids:
            for budget in BUDGETS:
                key=str(budget);commands[key]=old['commands']['N']
                prefix[key]=dict(actual_iterations=len(n.get('steps',[])),accepted_extra_steps=0,
                    extra_model_requests=0,extra_new_renders=0,extra_loop_seconds=0.,original_early_stop_reused=True)
        else:
            trajectory=HERE/'trajectories'/f'case_{case}.json'
            if trajectory.exists():
                saved=read(trajectory);commands=saved['commands'];prefix=saved['prefix'];reconstruction=saved['reconstruction'];error=saved.get('model_error')
            else:
                k=old_details(case,'K',archived_selection)
                physical=[s for s in k['steps'] if 'solver' in s]
                last=physical[-1]['cycle'] if physical and physical[-1]['accepted'] else None
                if last is None:raise BaselineAlignmentError('Active original N has no last K response')
                source=('development_k_adaptive8' if case in archived_selection['development'] else
                        'paired32_k_adaptive8' if case in archived_selection['confirmation'] else 'evaluation500_k_adaptive8')
                camera.begin_case(case);calls0=camera.calls;step2={};history=[]
                images,_=blocks.read('joint',case//25)
                def checkpoint(step,command,steps,seconds,requests,fd_calls,initial,cost,initial_secant_error):
                    if step==2:
                        difference=float(np.max(np.abs(command-np.asarray(old['commands']['N']))))
                        if difference>1e-8:
                            raise BaselineAlignmentError(f'Case{case} original2step endpoint changed by {difference}; stop instead of mixing baselines')
                        step2.update(seconds=seconds,requests=requests,renders=camera.calls-calls0,endpoint_difference=difference)
                        reconstruction.update(seconds=seconds,new_renders=camera.calls-calls0)
                    if step in BUDGETS:
                        key=str(step);commands[key]=command.tolist()
                        prefix[key]=dict(actual_iterations=step,accepted_extra_steps=sum(s['accepted'] for s in steps[2:]),
                            extra_model_requests=requests-step2['requests'],extra_new_renders=camera.calls-calls0-step2['renders'],
                            extra_loop_seconds=seconds-step2['seconds'],final_image_cost=cost,
                            baseline_endpoint_difference=step2['endpoint_difference'])
                        common.atomic(HERE/'trajectories'/f'partial_{case}.json',dict(case=case,commands=commands,prefix=prefix))
                    state('original_N_continuation',completed=len(rows),total=150,case=case,iteration=step,max_iteration=16,
                          active_cases_total=len(active_ids))
                    if stopped():raise PauseRequested()
                state('reconstructing_N_state',completed=len(rows),total=150,case=case)
                result=refiner.predict(images[case%25],-np.asarray(old['commands']['network']),
                    -np.asarray(old['commands']['K']),case,source,last,
                    k.get('final_center_limit',k.get('center_limit',.01)),steps=16,extra=4,on_step=checkpoint)
                if not step2:raise BaselineAlignmentError(f'Case{case} failed while rebuilding original N state: {result.get("error",result.get("stop_reason"))}')
                error=result.get('error');history=result['steps'];end=result['command'].tolist()
                for budget in BUDGETS:
                    key=str(budget)
                    if key not in commands:
                        commands[key]=end.copy()
                        prefix[key]=dict(actual_iterations=len(history),accepted_extra_steps=sum(s['accepted'] for s in history[2:]),
                            extra_model_requests=result['actual_camera_calls']-step2['requests'],
                            extra_new_renders=camera.calls-calls0-step2['renders'],
                            extra_loop_seconds=result['seconds']-step2['seconds'],
                            final_image_cost=result.get('final_image_cost'),stopped_early=True)
                common.atomic(trajectory,dict(case=case,commands=commands,prefix=prefix,reconstruction=reconstruction,
                    steps=history,model_error=error,stop_reason=result['stop_reason'],state_reconstruction=step2))
        # Four commands are fixed before hidden poses enter original scoring.
        commands['2']=old['commands']['N'];scores={'2':old['metrics']['N'].copy()}
        state('scoring',completed=len(rows),total=150,case=case)
        for budget in BUDGETS[1:]:
            key=str(budget);equal=next((b for b in scores if np.array_equal(commands[key],commands[b])),None)
            if equal is not None:scores[key]=scores[equal].copy()
            else:
                values,valid=metric.score(hidden[case],np.asarray(commands[key]));scores[key]=score_dict(values[0],valid[0])
        row=dict(case=case,commands=commands,metrics=scores,prefix=prefix,reconstruction=reconstruction,
            continued=case in active_ids,active_seconds=common.clock()-tick,model_error=error)
        common.atomic(path,row);rows.append(row);report=summary(rows);common.atomic(HERE/'partial_summary.json',report)
        active_times=[r['active_seconds'] for r in rows if r['continued']][-10:]
        remaining_active=len(active_ids)-sum(r['continued'] for r in rows)
        eta=float(np.mean(active_times))*remaining_active if active_times else None
        state('case_completed',completed=len(rows),total=150,last_case=case,remaining_active_cases=remaining_active,
              estimated_remaining_seconds=eta)
        print(json.dumps(dict(case=case,completed=len(rows),continued=case in active_ids,
            fullfield={key:report['metrics'][key]['fullfield'] for key in ('2','4','8','16')},
            seconds=row['active_seconds'],eta_seconds=eta)),flush=True)
    report=summary(rows);common.atomic(HERE/'result.json',report)
    data=dict(index=np.asarray([r['case'] for r in rows]))
    for budget in BUDGETS:
        key=str(budget);data[f'command_{key}']=np.asarray([r['commands'][key] for r in rows])
        data[f'fields_{key}']=np.asarray([r['metrics'][key]['fields'] for r in rows],dtype=float)
        data[f'valid_{key}']=np.asarray([r['metrics'][key]['valid'] for r in rows])
    np.savez_compressed(HERE/'predictions150.npz',**data)
    lines=['Original N iteration-budget comparison on fixed150; generated results await assistant interpretation.']
    for budget in BUDGETS:
        key=str(budget);m=report['metrics'][key]
        lines.append(f"N{budget}: fullfield{m['fullfield']}/150;center{m['center']};invalid{m['invalid']};center_mean{m['center_mean']:.9f};edge_mean{m['edge_mean']:.9f}")
    lines.append('Only iteration cap changed. No automatic500 or deployment.')
    (HERE/'GENERATED_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    state('computations_completed',completed=150,total=150,assistant_interpretation_pending=True,automatic_full500=False)
    print(json.dumps({k:v for k,v in report.items() if k!='costs'},indent=2),flush=True)


if __name__=='__main__':
    with (HERE/'run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:main()
        except PauseRequested:state('paused',reason='STOP_REQUESTED.json observed between original steps')
        except BaseException:
            state('failed',error=traceback.format_exc());raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
