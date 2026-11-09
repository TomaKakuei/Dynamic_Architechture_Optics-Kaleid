"""Original Olympus500, bounded8 model steps, resumable after each step/case.

Reuse the eight completed pilot commands. Evaluate all remaining rows once.
No training, new measurements, hidden-pose inference, hash checks or replay.
"""
from support import *
from camera_inverse import CameraModel
from candidate import refine
from diff_metrics import DiffMetrics
import json,os,traceback,ctypes

OUT=HERE/'evaluation500'


def summarize(rows):
    result=dict(cases=len(rows),target_cases=500,original_center_threshold=.07,
        fullfield_definition='Center ABS; eight edges signed phase minus nominal phase on common pupil, followed by PTTD removal; all nine below .07 waves and original validity',
        acquired_new_packets=0,training_updates=0,independent_review=False,
        automatic_deployment=False,active_seconds=sum(r['active_seconds'] for r in rows),
        field_names=['center']+['edge'+str(k) for k in range(1,9)])
    masks={}
    for key in ('baseline','candidate'):
        values=np.array([r[key]['mixed_wrms'] for r in rows]);valid=np.array([r[key]['valid'] for r in rows])
        passed=valid&(values<.07).all(1);center=valid&(values[:,0]<.07)
        result[key]=dict(center_pass=int(center.sum()),center_percent=float(center.mean()*100),
            fullfield_pass=int(passed.sum()),fullfield_percent=float(passed.mean()*100),
            invalid=int((~valid).sum()),
            mean_center_all_finite=float(np.nanmean(values[:,0])),
            mean_edge_diff_all_finite=float(np.nanmean(values[:,1:])),
            mean_fields_all_finite=np.nanmean(values,axis=0).tolist())
        masks[key]=(passed,center)
    b,c=masks['baseline'][0],masks['candidate'][0]
    result.update(fullfield_gained=int((~b&c).sum()),fullfield_lost=int((b&~c).sum()),
                  retained_pilot_cases=sum(r['source']=='retained_pilot' for r in rows),
                  assistant_reviewed=False)
    return result


def main():
    OUT.mkdir(exist_ok=True)
    if (HERE/'STOP_REQUESTED.json').exists():
        print('Stopped by user. Explicit reauthorization is required before resuming.',flush=True);return
    if (OUT/'result.json').exists():
        print('Already complete; no repeated evaluation.',flush=True);return
    # Prevent system idle sleep only during this process; never alter power settings.
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    with np.load(SOURCE/'training/selected_validation.npz') as z:old={k:z[k].copy() for k in z.files}
    protocol=dict(created=common.stamp(),pid=os.getpid(),cases=500,case_ids=old['index'].tolist(),
        frozen_weights=str(SOURCE/'training/candidate.pt'),neural_output='Reuse selected_validation.npz commands',
        packets='Original whole-assembly noisy nine-field128x128 camera packet per case; zero new acquisitions',
        candidate='Full camera-image objective; at most8 model-only Gauss-Newton steps; stop if no accepted step',
        ridge=1e-4,coordinate_step_cap=.25,line_gains=[1,.5,.25],estimate_bounds=[-1,1],
        numerical_camera='1024 pupil/2048FFT; degree24; original diffuser, relay, ADU model; boundary frozen locally in Jacobian and refreshed in objective',
        original_reference_camera='2048 pupil/4096FFT; recorded adaptive degree24/28/32/36',
        reuse_pilot=True,no_truth_in_inference=True,independent_review=False,digest_comparisons=False,
        automatic_followup=False,automatic_deployment=False,notification_configured=False)
    if not (OUT/'protocol.json').exists():common.atomic(OUT/'protocol.json',protocol)
    state('evaluation500_initializing',pid=os.getpid(),total=500)
    camera=CameraModel();metrics=DiffMetrics();blocks=Blocks()
    pilot=json.loads((HERE/'pilot/summary.json').read_text());pilot={r['case']:r for r in pilot['rows']}
    diffs=json.loads((HERE/'pilot/diff_metrics.json').read_text());diffs={r['case']:r for r in diffs['rows']}
    completed=[];tick=common.clock();this_run_count=0
    # Prepopulate saved pilot results; no camera inverse or optical scoring repeated.
    for case,p in pilot.items():
        dest=OUT/f'case_{case}.json'
        if dest.exists():continue
        d=diffs[case]
        row=dict(case=case,source='retained_pilot',command=p['command'],steps=p['steps'],
            active_seconds=p['active_seconds'],stop_reason='original_eight_step_pilot',
            baseline=dict(mixed_wrms=d['before'],valid=d['before_valid']),
            candidate=dict(mixed_wrms=d['after8'],valid=d['after_valid']))
        common.atomic(dest,row)
    for slot,case in enumerate(old['index']):
        case=int(case);dest=OUT/f'case_{case}.json'
        if dest.exists():completed.append(json.loads(dest.read_text()));continue
        cached=list(OUT.glob('case_*.json'));done=len(cached)
        state('evaluation500',pid=os.getpid(),case=case,completed_cases=done,total=500)
        image,_=blocks.read('joint',case//25);image=image[case%25]
        def callback(step):state('evaluation500',pid=os.getpid(),case=case,completed_cases=done,total=500,model_step=step,max_model_steps=8)
        refined=refine(camera,image,-old['command'][slot],OUT/f'partial_{case}.json',callback)
        command=-np.array(refined['estimate'])
        # Truth first enters here, after the final command has been decided.
        score_tick=common.clock()
        fields,valid=metrics.score(np.repeat(old['initial'][slot][None],2,axis=0),np.stack([old['command'][slot],command]))
        row=dict(case=case,source='new_inference',command=command.tolist(),steps=refined['steps'],
            stop_reason=refined['stop_reason'],active_seconds=refined['active_seconds']+common.clock()-score_tick,
            baseline=dict(mixed_wrms=fields[0].tolist(),valid=bool(valid[0])),
            candidate=dict(mixed_wrms=fields[1].tolist(),valid=bool(valid[1])))
        common.atomic(dest,row);completed.append(row);this_run_count+=1
        newdone=done+1;avg=(common.clock()-tick)/this_run_count
        state('evaluation500',pid=os.getpid(),case=case,completed_cases=newdone,total=500,
              active_seconds_this_process=common.clock()-tick,mean_seconds_per_new_case=avg,
              estimated_remaining_active_seconds=avg*(500-newdone))
        print(json.dumps(dict(case=case,done=newdone,seconds=row['active_seconds'],steps=len(row['steps']),
            before_valid=row['baseline']['valid'],after_valid=row['candidate']['valid'],
            before_max=max(row['baseline']['mixed_wrms']),after_max=max(row['candidate']['mixed_wrms']),
            remaining_seconds=avg*(500-newdone))),flush=True)
    result=summarize(completed)
    common.atomic(OUT/'result.json',result)
    np.savez_compressed(OUT/'predictions.npz',index=np.array([r['case'] for r in completed]),
        command=np.array([r['command'] for r in completed]),
        baseline_mixed=np.array([r['baseline']['mixed_wrms'] for r in completed]),
        candidate_mixed=np.array([r['candidate']['mixed_wrms'] for r in completed]),
        baseline_valid=np.array([r['baseline']['valid'] for r in completed]),
        candidate_valid=np.array([r['candidate']['valid'] for r in completed]))
    state('evaluation500_completed',pid=os.getpid(),completed_cases=500,total=500,
          result=str(OUT/'result.json'),assistant_reviewed=False)
    print(json.dumps(result,indent=2),flush=True)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    try:main()
    except BaseException:
        state('evaluation500_failed',pid=os.getpid(),error=traceback.format_exc());raise
