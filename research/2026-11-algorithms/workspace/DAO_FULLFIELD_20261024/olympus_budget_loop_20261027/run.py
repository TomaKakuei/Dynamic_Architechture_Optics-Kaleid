"""Resume saved N; screen on old44, then conditionally extend to original500."""
from continuation import *
import json, os, ctypes, traceback, msvcrt
from secant import Blocks, DiffMetrics, poses
FULL = HERE.parent/'olympus_full500_20261027'
DROOT = HERE.parent/'olympus_lowcost_refiner_20261026'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def state(stage, **kw):
    common.atomic(HERE/'state.json', dict(stage=stage, updated=common.stamp(), pid=os.getpid(), **kw))


def details(case, route):
    path = FULL/'stages'/route/f'case_{case}.json'
    if path.exists():
        return read(path)['details']
    source = ('development' if case in SELECTION['development'] else 'paired32')
    root, suffix = (KROOT, '_k_adaptive8') if route == 'K' else (NROOT, '_n_expand4')
    return read(root/(source+suffix)/f'case_{case}.json')


SELECTION = read(DROOT/'selection.json')
SCREEN = SELECTION['development']+SELECTION['confirmation']
ALL_IDS = list(range(7000, 7500))


def summary(rows, total):
    result = dict(completed=len(rows), total=total, complete=len(rows)==total, metrics={})
    flags = {}
    for route in ('network', 'D', 'K', 'N', 'candidate'):
        fields = np.asarray([r['metrics'][route]['fields'] for r in rows], dtype=float)
        valid = np.asarray([r['metrics'][route]['valid'] for r in rows], dtype=bool)
        full = valid & np.isfinite(fields).all(1) & (fields < .07).all(1)
        center = valid & np.isfinite(fields[:, 0]) & (fields[:, 0] < .07)
        flags[route] = (full, center, valid)
        result['metrics'][route] = dict(fullfield=int(full.sum()), center=int(center.sum()), invalid=int((~valid).sum()),
            center_mean=float(np.nanmean(fields[:, 0])), edge_mean=float(np.nanmean(fields[:, 1:])))
    a, b = flags['N'], flags['candidate']
    result.update(gained=[rows[i]['case'] for i in np.where(~a[0]&b[0])[0]],
        lost=[rows[i]['case'] for i in np.where(a[0]&~b[0])[0]],
        center_gained=int((~a[1]&b[1]).sum()), center_lost=int((a[1]&~b[1]).sum()),
        new_invalid=int((a[2]&~b[2]).sum()), fixed_invalid=int((~a[2]&b[2]).sum()),
        model_errors=[r['case'] for r in rows if 'error' in r['details']],
        incremental_seconds=sum(r['active_seconds'] for r in rows),
        camera_calls=sum(r['details']['actual_camera_calls'] for r in rows),
        finite_difference_probes=sum(r['details']['new_finite_difference_probes'] for r in rows),
        timing_scope='Incremental continuation and offline scoring after saved N; excludes original D/K/N, neural inference and initialization',
        truth_in_inference=False, new_acquisitions=0, training_updates=0)
    return result


def screen_gate(result):
    old, new = result['metrics']['N'], result['metrics']['candidate']
    return bool(new['fullfield'] > old['fullfield'] and not result['lost']
                and new['center'] >= old['center'] and result['new_invalid']==0
                and not result['model_errors'])


def main():
    if (HERE/'result.json').exists():
        print((HERE/'result.json').read_text()); return
    if (HERE/'screen_result.json').exists() and not screen_gate(read(HERE/'screen_result.json')):
        state('screen_failed_no_full_run', result=read(HERE/'screen_result.json')); return
    if (HERE/'STOP_REQUESTED.json').exists():
        state('paused'); return
    (HERE/'cases').mkdir(exist_ok=True)
    protocol = dict(screen_ids=SCREEN, all_ids=ALL_IDS,
        scope='Original validation, old12 development plus old32 comparison screen, not fresh confirmation',
        baseline='Completed frozen N, 373/500',
        continuation='Four more model-only steps from exact saved N endpoints; twelve transferred responses plus four fresh central directions',
        total_N_steps_max=6, search_rank=16, gains=[1., .5, .25, .125, .0625],
        symmetric_probe_h=.005, max_new_probes_per_active_case=8,
        eligibility='Old N actually had a response/step; based only on solver history, never recovery labels',
        prior_radius=.2, step_limit=.05, center_limit='Frozen K terminal nominal proxy; not actual endpoint guarantee',
        J_transfer='One aggregate secant from saved K endpoint to N; N final Jacobian was not stored',
        screen_gate='Full recovery increases; zero prior N full successes lost; center count not lower; zero new invalid or model error',
        conditional_full500=True, automatic_deployment=False, truth_in_inference=False,
        new_acquisitions=0, training_updates=0, scoring='Original centerABS + signed edgeDIFF all9<.07 plus original validity')
    common.atomic(HERE/'protocol.json', protocol)
    state('initializing', completed=len(list((HERE/'cases').glob('case_*.json'))), total=500)
    tick=common.clock(); refiner=BudgetRefiner(); blocks=Blocks(); metric=DiffMetrics(); hidden_poses=poses()
    common.atomic(HERE/'initialization.json', dict(seconds=common.clock()-tick, pid=os.getpid()))
    rows=[]; order=SCREEN+[i for i in ALL_IDS if i not in SCREEN]
    for case in order:
        if (HERE/'STOP_REQUESTED.json').exists():
            state('paused', completed=len(rows), total=500); return
        output=HERE/'cases'/f'case_{case}.json'
        if output.exists():
            rows.append(read(output))
        else:
            old=read(FULL/'cases'/f'case_{case}.json'); k=details(case, 'K'); n=details(case, 'N')
            physical=[s for s in k['steps'] if 'solver' in s]
            last=physical[-1]['cycle'] if physical and physical[-1]['accepted'] else None
            source=('development_k_adaptive8' if case in SELECTION['development'] else
                    'paired32_k_adaptive8' if case in SELECTION['confirmation'] else 'evaluation500_k_adaptive8')
            eligible=bool(n.get('steps'))
            state('evaluating_screen' if len(rows)<44 else 'evaluating_full500', case=case, completed=len(rows), total=500)
            tick=common.clock()
            images,_=blocks.read('joint',case//25)
            result=refiner.predict(images[case%25], -np.asarray(old['commands']['network']),
                -np.asarray(old['commands']['N']), -np.asarray(old['commands']['K']), case, source, last,
                k.get('final_center_limit', k.get('center_limit', .01)), eligible)
            command=result.pop('command'); scores=old['metrics'].copy()
            if np.array_equal(command, np.asarray(old['commands']['N'])):
                scores['candidate']=scores['N'].copy()
            else:
                values,valid=metric.score(hidden_poses[case],command)
                values=values[0]; finite=np.isfinite(values)
                scores['candidate']=dict(fields=[float(v) if np.isfinite(v) else None for v in values],
                                         valid=bool(valid[0] and finite.all()),nonfinite_fields=int((~finite).sum()))
            row=dict(case=case,command=command.tolist(),metrics=scores,details=result,active_seconds=common.clock()-tick)
            common.atomic(output,row); rows.append(row)
            print(json.dumps(dict(case=case,completed=len(rows),calls=result['actual_camera_calls'],
                accepted_steps=sum(s['accepted'] for s in result['steps']),seconds=row['active_seconds'],
                old_full=bool(scores['N']['valid'] and max(scores['N']['fields'])<.07),
                new_full=bool(scores['candidate']['valid'] and all(v is not None and v<.07 for v in scores['candidate']['fields'])),
                error=result.get('error'))),flush=True)
        total=44 if len(rows)<=44 else 500
        report=summary(rows,total); common.atomic(HERE/'partial_summary.json',report)
        if len(rows)==44:
            common.atomic(HERE/'screen_result.json',report)
            if not screen_gate(report):
                state('screen_failed_no_full_run',completed=44,total=500,assistant_interpretation_pending=True,result=report)
                print(json.dumps(dict(screen_passed=False,result=report)),flush=True); return
            common.atomic(HERE/'screen_accepted.json',dict(updated=common.stamp(),result=report,
                                                         frozen_protocol='protocol.json'))
            print(json.dumps(dict(screen_passed=True,result=report)),flush=True)
        recent=[r['active_seconds'] for r in rows[-20:]]
        state('case_completed', completed=len(rows), total=44 if len(rows)<44 else 500,
              estimated_remaining_seconds=float(np.mean(recent))*((44 if len(rows)<44 else 500)-len(rows)))
    rows=sorted(rows,key=lambda r:r['case']); report=summary(rows,500)
    common.atomic(HERE/'result.json',report)
    np.savez_compressed(HERE/'predictions500.npz',index=np.asarray([r['case'] for r in rows]),
        command=np.asarray([r['command'] for r in rows]),
        fields=np.asarray([r['metrics']['candidate']['fields'] for r in rows],dtype=float),
        valid=np.asarray([r['metrics']['candidate']['valid'] for r in rows]))
    state('computations_completed',completed=500,total=500,assistant_interpretation_pending=True,
          automatic_deployment=False,result='result.json')


if __name__=='__main__':
    with (HERE/'run.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1): lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:main()
        except BaseException:
            state('failed',error=traceback.format_exc());raise
        finally:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)
