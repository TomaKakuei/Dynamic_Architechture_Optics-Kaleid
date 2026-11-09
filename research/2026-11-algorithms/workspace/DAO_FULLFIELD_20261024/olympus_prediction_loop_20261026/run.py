from context import *
from predictor import PredictorLoop
import argparse,traceback


def report(rows):
    out=dict(cases=len(rows));flags={}
    for name in ('network','previous','candidate'):
        values=np.array([r[name]['fields'] for r in rows]);valid=np.array([r[name]['valid'] for r in rows])
        full=valid&(values<.07).all(1);center=valid&(values[:,0]<.07)
        flags[name]=(full,center,valid)
        out[name]=dict(fullfield_pass=int(full.sum()),center_pass=int(center.sum()),invalid=int((~valid).sum()),
            center_mean=float(np.nanmean(values[:,0])),edge_mean=float(np.nanmean(values[:,1:])))
    b,c=flags['previous'][0],flags['candidate'][0]
    out.update(gained_over_previous=int((~b&c).sum()),lost_previous=int((b&~c).sum()),
        center_gained=int((~flags['previous'][1]&flags['candidate'][1]).sum()),
        center_lost=int((flags['previous'][1]&~flags['candidate'][1]).sum()),
        new_invalid=int((flags['previous'][2]&~flags['candidate'][2]).sum()),
        additional_seconds=sum(r['seconds'] for r in rows),additional_model_calls=sum(r['actual_camera_calls'] for r in rows),
        additional_network_calls=sum(r['network_calls'] for r in rows),
        timing_scope='Incremental loop only; previous D outputs/model caches reused; not complete cold pipeline timing',
        observations_per_case=1,truth_used_in_inference=False,training_updates=0,automatic_deployment=False)
    return out


def main(args):
    directory=ROOT/args.tag;directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():print((directory/'result.json').read_text());return
    selection=json.loads((PREVIOUS/'selection.json').read_text());ids=selection[args.split]
    if args.mode in ('relinearized','adaptive'):
        from model_loop import ModelLoop
        loop=ModelLoop()
    else:loop=PredictorLoop()
    old=baseline();truth=poses();blocks=Blocks();metric=DiffMetrics();rows=[]
    common.atomic(directory/'protocol.json',dict(ids=ids,split=args.split,scope='Paired extension of saved D on previously observed cases; exploratory, not fresh confirmation',
        cycles=args.cycles,radius=args.radius,mode=args.mode,noise_stop=args.noise_stop,center_limit=.01,
        local_gains=[1,.5,1.5] if args.mode in ('relinearized','adaptive') else [1,.5,-.5],
        current_state_repredicted_each_cycle=True,measurement_packets=1,truth_in_inference=False,
        adaptive_center_trust=args.mode=='adaptive',max_center_limit=.04 if args.mode=='adaptive' else .01,
        prediction_agreement_threshold=.75 if args.mode=='adaptive' else None,
        reuses_first_linearization_cache=args.mode=='adaptive'))
    for case in ids:
        path=directory/f'case_{case}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        previous_tag='development_d8_center' if args.split=='development' else 'confirmation_d8_center'
        previous=json.loads((PREVIOUS/previous_tag/f'case_{case}.json').read_text())
        slot=int(np.where(old['index']==case)[0][0]);image,_=blocks.read('joint',case//25)
        state('prediction_loop',tag=args.tag,case=case,completed=len(rows),total=len(ids))
        result=loop.predict_loop(image[case%25],-old['command'][slot],-np.array(previous['command']),case,
            cycles=args.cycles,radius=args.radius,mode=args.mode,noise_stop=args.noise_stop)
        command=result.pop('command')
        if np.array_equal(command,np.array(previous['command'])):values=np.array(previous['candidate']['fields'])[None];valid=[previous['candidate']['valid']]
        else:values,valid=metric.score(truth[case],command)
        row=dict(case=case,command=command.tolist(),network=previous['baseline'],previous=previous['candidate'],
                 candidate=dict(fields=values[0].tolist(),valid=bool(valid[0])),**result)
        common.atomic(path,row);rows.append(row)
        print(json.dumps(dict(case=case,previous_pass=bool(row['previous']['valid'] and max(row['previous']['fields'])<.07),
            new_pass=bool(row['candidate']['valid'] and max(row['candidate']['fields'])<.07),valid=bool(valid[0]),
            seconds=row['seconds'],calls=row['actual_camera_calls'],cost_before=row.get('initial_image_cost'),cost_after=row.get('final_image_cost'))),flush=True)
    result=report(rows);common.atomic(directory/'result.json',result);state('trial_completed',tag=args.tag,result=result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--split',default='development',choices=['development','confirmation']);p.add_argument('--cycles',type=int,default=2);p.add_argument('--radius',type=float,default=.1);p.add_argument('--mode',choices=['neural','pattern','relinearized','adaptive'],default='neural');p.add_argument('--noise-stop',type=float,default=8.)
    try:main(p.parse_args())
    except BaseException:state('failed',error=traceback.format_exc());raise
