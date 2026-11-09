from secant import *
import argparse,json,os,traceback


def state(stage,**kw):common.atomic(ROOT/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))


def report(rows):
    result=dict(cases=len(rows));flags={}
    for name in ('network','lowcost','loop','previous','candidate'):
        v=np.array([r[name]['fields'] for r in rows]);valid=np.array([r[name]['valid'] for r in rows])
        full=valid&(v<.07).all(1);center=valid&(v[:,0]<.07);flags[name]=(full,center,valid)
        result[name]=dict(full=int(full.sum()),center=int(center.sum()),invalid=int((~valid).sum()),
            edge_mean=float(np.nanmean(v[:,1:])),center_mean=float(np.nanmean(v[:,0])))
    a,b=flags['previous'],flags['candidate']
    result.update(gained=[rows[i]['case'] for i in np.where(b[0]&~a[0])[0]],
        lost=[rows[i]['case'] for i in np.where(a[0]&~b[0])[0]],center_gained=int((b[1]&~a[1]).sum()),
        center_lost=int((a[1]&~b[1]).sum()),new_invalid=int((a[2]&~b[2]).sum()),
        additional_seconds=sum(r['seconds'] for r in rows),additional_camera_calls=sum(r['actual_camera_calls'] for r in rows),
        model_errors=[r['case'] for r in rows if 'error' in r],new_finite_difference_probes=sum(r.get('new_finite_difference_probes',0) for r in rows),
        timing='Incremental after saved K with prior responses; not cold total pipeline latency',
        training_updates=0,new_acquisitions=0,truth_in_inference=False)
    return result


def main(args):
    directory=ROOT/args.tag;directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():print((directory/'result.json').read_text());return
    ids=json.loads((PREVIOUS/'selection.json').read_text())[args.split]
    source='development_k_adaptive8' if args.split=='development' else 'paired32_k_adaptive8'
    common.atomic(directory/'protocol.json',dict(ids=ids,source=source,scope='Previously observed exploratory paired comparison',
        steps=args.steps,fixed_original_K_subspace=args.extra==0,extra_directions=args.extra,
        update='Rank-one secant using synthetic model-image change; optional current central responses outside original8',
        max_new_finite_difference_probes_per_active_case=2*args.extra,noise_stop='Predicted gain>2*rank/128^2 instead of global image cost gate',
        eligibility='Prior K actually accepted a step; no label-based routing',
        center_limit='Frozen at K terminal value',radius=.2,step_max=.05,truth_in_inference=False))
    refiner=SecantRefiner();old=baseline();truth=poses();blocks=Blocks();metric=DiffMetrics();rows=[]
    for case in ids:
        if (ROOT/'STOP_REQUESTED.json').exists():state('stopped',tag=args.tag,completed=len(rows));return
        path=directory/f'case_{case}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        k=json.loads((BASE/source/f'case_{case}.json').read_text())
        physical_steps=[s for s in k['steps'] if 'solver' in s]
        last=physical_steps[-1]['cycle'] if physical_steps and physical_steps[-1]['accepted'] else None
        slot=int(np.where(old['index']==case)[0][0]);image,_=blocks.read('joint',case//25)
        state('secant_refinement',tag=args.tag,case=case,completed=len(rows),total=len(ids))
        result=refiner.predict(image[case%25],-old['command'][slot],-np.array(k['command']),case,source,last,
            k.get('final_center_limit',k.get('center_limit',.01)),args.steps,args.extra)
        command=result.pop('command')
        if np.array_equal(command,np.array(k['command'])):values=np.array(k['candidate']['fields'])[None];valid=[k['candidate']['valid']]
        else:values,valid=metric.score(truth[case],command)
        row=dict(case=case,command=command.tolist(),network=k['network'],lowcost=k['lowcost'],loop=k['loop'],previous=k['candidate'],
            candidate=dict(fields=values[0].tolist(),valid=bool(valid[0])),**result)
        common.atomic(path,row);rows.append(row)
        print(json.dumps(dict(case=case,full=bool(valid[0] and max(values[0])<.07),valid=bool(valid[0]),
            worst=float(max(values[0])),seconds=result['seconds'],calls=result['actual_camera_calls'],
            cost_before=result.get('initial_image_cost'),cost_after=result.get('final_image_cost'),error=result.get('error'))),flush=True)
    result=report(rows);common.atomic(directory/'result.json',result);state('trial_completed',tag=args.tag,result=result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--split',choices=['development','confirmation'],default='development');p.add_argument('--steps',type=int,default=2);p.add_argument('--extra',type=int,choices=[0,4],default=0)
    try:main(p.parse_args())
    except BaseException:state('failed',error=traceback.format_exc());raise
