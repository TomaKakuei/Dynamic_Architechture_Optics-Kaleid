from bridge import *
from refiner import Refiner
import argparse,traceback


def aggregate(rows):
    out=dict(cases=len(rows),scope='New four-epoch network plus same-packet model refinement; offline labels only score decided commands')
    flags={}
    for name in ('baseline','candidate'):
        a=np.array([r[name]['fields'] for r in rows]);v=np.array([r[name]['valid'] for r in rows])
        passed=v&(a<.07).all(1);center=v&(a[:,0]<.07);flags[name]=(passed,center,v)
        out[name]=dict(fullfield_pass=int(passed.sum()),center_pass=int(center.sum()),invalid=int((~v).sum()),
            center_mean_all_finite=float(np.nanmean(a[:,0])),edge_mean_all_finite=float(np.nanmean(a[:,1:])))
    b,c=flags['baseline'][0],flags['candidate'][0]
    out.update(gained=int((~b&c).sum()),lost=int((b&~c).sum()),new_invalid=int((flags['baseline'][2]&~flags['candidate'][2]).sum()),
        fixed_invalid=int((~flags['baseline'][2]&flags['candidate'][2]).sum()),
        active_seconds=sum(r['active_seconds'] for r in rows),
        mean_seconds_per_case=float(np.mean([r['active_seconds'] for r in rows])),
        actual_camera_calls=sum(r['actual_camera_calls'] for r in rows),
        note='Elapsed includes scoring; cache reuse is explicitly counted. Per-candidate cold cost requires logical model call counts.',
        training_updates=0,acquired_packets=0,automatic_deployment=False)
    return out


def run(args):
    dest=RUN/args.tag;dest.mkdir(exist_ok=True)
    if (dest/'result.json').exists():print((dest/'result.json').read_text());return
    selection=json.loads((RUN/'selection.json').read_text());ids=selection[args.split]
    if args.limit:ids=ids[:args.limit]
    common.atomic(dest/'protocol.json',dict(tag=args.tag,split=args.split,ids=ids,rank=args.rank,steps=args.steps,
        radius=args.radius,step_radius=args.step_radius,prior_strength=args.prior_strength,basis_mode=args.basis_mode,center_limit=args.center_limit,network=str(REPAIR/'training/last.pt'),
        camera='Original2048pupil/4096FFT; full phase/carrier/aperture/normalization response in retained subspace',
        inference_truth=False,noise_stop_cost=9,selection_scope=selection[args.split+'_scope']))
    old=baseline();q=poses();blocks=Blocks();refiner=Refiner(args.basis_mode);metrics=DiffMetrics();rows=[]
    for case in ids:
        output=dest/f'case_{case}.json'
        if output.exists():rows.append(json.loads(output.read_text()));continue
        slot=int(np.where(old['index']==case)[0][0]);image,_=blocks.read('joint',case//25)
        state('refiner_trial',tag=args.tag,case=case,completed=len(rows),total=len(ids),rank=args.rank)
        tick=common.clock()
        result=refiner.predict(image[case%25],-old['command'][slot],case,rank=args.rank,steps=args.steps,
            radius=args.radius,step_radius=args.step_radius,prior_strength=args.prior_strength,center_limit=args.center_limit)
        command=result.pop('command')
        if np.array_equal(command,old['command'][slot]):
            new=old['mixed_wrms'][slot][None];valid=old['valid'][slot:slot+1]
        else:new,valid=metrics.score(q[case],command)
        row=dict(case=case,command=command.tolist(),baseline=dict(fields=old['mixed_wrms'][slot].tolist(),valid=bool(old['valid'][slot])),
                 candidate=dict(fields=new[0].tolist(),valid=bool(valid[0])),**result)
        row['active_seconds']=common.clock()-tick
        common.atomic(output,row);rows.append(row)
        print(json.dumps(dict(case=case,baseline_pass=bool(row['baseline']['valid'] and max(row['baseline']['fields'])<.07),
            candidate_pass=bool(row['candidate']['valid'] and max(row['candidate']['fields'])<.07),valid=bool(valid[0]),
            seconds=row['active_seconds'],calls=row['actual_camera_calls'])),flush=True)
    report=aggregate(rows);common.atomic(dest/'result.json',report);state('trial_complete',tag=args.tag,result=report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--split',default='development',choices=['development','confirmation']);p.add_argument('--rank',type=int,default=4);p.add_argument('--steps',type=int,default=1);p.add_argument('--radius',type=float,default=.1);p.add_argument('--step-radius',type=float,default=.05);p.add_argument('--prior-strength',type=float,default=1.);p.add_argument('--limit',type=int);p.add_argument('--basis-mode',default='goal',choices=['goal','hybrid'])
    p.add_argument('--center-limit',type=float,default=0.)
    try:run(p.parse_args())
    except BaseException:state('failed',error=traceback.format_exc());raise
