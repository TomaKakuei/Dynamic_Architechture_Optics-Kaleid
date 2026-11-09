from solver import *
import argparse,traceback,os


def state(stage,**kw):common.atomic(TASK/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))


def report(rows):
    result=dict(cases=len(rows));flags={}
    for route in ('network','lowcost','loop','candidate'):
        v=np.array([r[route]['fields'] for r in rows]);valid=np.array([r[route]['valid'] for r in rows])
        full=valid&(v<.07).all(1);center=valid&(v[:,0]<.07);flags[route]=(full,center,valid)
        result[route]=dict(full=int(full.sum()),center=int(center.sum()),invalid=int((~valid).sum()),
            edge_mean=float(np.nanmean(v[:,1:])),center_mean=float(np.nanmean(v[:,0])))
    a,b=flags['loop'],flags['candidate']
    result.update(gained_over_loop=[rows[i]['case'] for i in np.where(b[0]&~a[0])[0]],
        lost_loop=[rows[i]['case'] for i in np.where(a[0]&~b[0])[0]],center_gained=int((b[1]&~a[1]).sum()),
        center_lost=int((a[1]&~b[1]).sum()),new_invalid=int((a[2]&~b[2]).sum()),
        additional_seconds=sum(r['seconds'] for r in rows),additional_camera_calls=sum(r['actual_camera_calls'] for r in rows),
        timing='Incremental cache-assisted research time; not cold total pipeline latency',
        training_updates=0,new_acquisitions=0,truth_in_inference=False)
    return result


def main(args):
    directory=TASK/args.tag;directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():print((directory/'result.json').read_text());return
    ids=json.loads((PREVIOUS/'selection.json').read_text())[args.split]
    common.atomic(directory/'protocol.json',dict(ids=ids,scope='Previously observed exploratory paired comparison',
        cycles=args.cycles,fd=args.fd,rank=args.rank,radius=.2,center_limit=args.center_limit,adaptive=args.adaptive,
        solve='SLSQP convex local quadratic with joint box and center ellipsoid constraints',
        posthoc_radial_projection=False,local_gains=[1,.5,.25],truth_in_inference=False,new_acquisitions=0))
    refiner=ConstrainedRefiner(args.tag,args.fd,args.rank)
    old=baseline();truth=poses();blocks=Blocks();metric=DiffMetrics();rows=[]
    for case in ids:
        if (TASK/'STOP_REQUESTED.json').exists():
            state('paused',tag=args.tag,completed=len(rows),total=len(ids));return
        path=directory/f'case_{case}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        tag='development_f_model_loop' if args.split=='development' else 'paired32_f_model_loop'
        f=json.loads((OLD/tag/f'case_{case}.json').read_text())
        dtag='development_d8_center' if args.split=='development' else 'confirmation_d8_center'
        d=json.loads((PREVIOUS/dtag/f'case_{case}.json').read_text())
        slot=int(np.where(old['index']==case)[0][0]);image,_=blocks.read('joint',case//25)
        state('constrained_refinement',tag=args.tag,case=case,completed=len(rows),total=len(ids))
        result=refiner.predict(image[case%25],-old['command'][slot],-np.array(d['command']),case,
            cycles=args.cycles,center_limit=args.center_limit,adaptive=args.adaptive)
        command=result.pop('command')
        if np.array_equal(command,np.array(d['command'])):v=np.array(d['candidate']['fields'])[None];valid=[d['candidate']['valid']]
        else:v,valid=metric.score(truth[case],command)
        row=dict(case=case,command=command.tolist(),network=f['network'],lowcost=f['previous'],loop=f['candidate'],
            candidate=dict(fields=v[0].tolist(),valid=bool(valid[0])),**result)
        common.atomic(path,row);rows.append(row)
        print(json.dumps(dict(case=case,pass_full=bool(valid[0] and max(v[0])<.07),valid=bool(valid[0]),
            worst=float(max(v[0])),seconds=result['seconds'],calls=result['actual_camera_calls'],
            cost_before=result.get('initial_image_cost'),cost_after=result.get('final_image_cost'))),flush=True)
    result=report(rows);common.atomic(directory/'result.json',result);state('trial_completed',tag=args.tag,result=result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--split',choices=['development','confirmation'],default='development')
    p.add_argument('--fd',choices=['legacy','central'],default='legacy');p.add_argument('--rank',type=int,default=8)
    p.add_argument('--cycles',type=int,default=3);p.add_argument('--center-limit',type=float,default=.01);p.add_argument('--adaptive',action='store_true')
    try:main(p.parse_args())
    except BaseException:state('failed',error=traceback.format_exc());raise
