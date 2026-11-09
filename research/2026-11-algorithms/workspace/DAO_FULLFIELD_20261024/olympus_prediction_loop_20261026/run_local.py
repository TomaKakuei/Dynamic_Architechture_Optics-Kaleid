from context import *
from local_search import QuadraticLocalSearch
from run import report
import argparse,traceback


def main(args):
    directory=ROOT/args.tag;directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():print((directory/'result.json').read_text());return
    ids=json.loads((PREVIOUS/'selection.json').read_text())[args.split]
    source=ROOT/args.source
    if not (source/'result.json').exists():raise RuntimeError('Source trial must be complete before local search')
    common.atomic(directory/'protocol.json',dict(ids=ids,source=args.source,
        scope='Previously observed cases, exploratory paired extension after F',axes=[8,9],
        bracket=.025,tangent_to_active_center=True,quadratic_interpolation=True,
        radius=.2,center_limit=.01,noise_stop=9,measurement_packets=1,truth_in_inference=False))
    loop=QuadraticLocalSearch();old=baseline();truth=poses();blocks=Blocks();metric=DiffMetrics();rows=[]
    for case in ids:
        path=directory/f'case_{case}.json'
        if path.exists():rows.append(json.loads(path.read_text()));continue
        previous=json.loads((source/f'case_{case}.json').read_text())
        slot=int(np.where(old['index']==case)[0][0]);images,_=blocks.read('joint',case//25)
        state('quadratic_local_search',tag=args.tag,case=case,completed=len(rows),total=len(ids))
        result=loop.predict_local(images[case%25],-old['command'][slot],-np.array(previous['command']),case)
        command=result.pop('command')
        if np.array_equal(command,np.array(previous['command'])):
            values=np.array(previous['candidate']['fields'])[None];valid=[previous['candidate']['valid']]
        else:values,valid=metric.score(truth[case],command)
        row=dict(case=case,command=command.tolist(),network=previous['network'],lowcost=previous['previous'],
            previous=previous['candidate'],candidate=dict(fields=values[0].tolist(),valid=bool(valid[0])),**result)
        common.atomic(path,row);rows.append(row)
        print(json.dumps(dict(case=case,previous_pass=bool(row['previous']['valid'] and max(row['previous']['fields'])<.07),
            new_pass=bool(row['candidate']['valid'] and max(row['candidate']['fields'])<.07),valid=bool(valid[0]),
            seconds=row['seconds'],calls=row['actual_camera_calls'],cost_before=row.get('initial_image_cost'),
            cost_after=row.get('final_image_cost'))),flush=True)
    result=report(rows);result['previous_route']='F relinearized loop'
    result['timing_scope']='Additional G search only after cached/saved F; not complete cold pipeline timing'
    common.atomic(directory/'result.json',result);state('trial_completed',tag=args.tag,result=result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--source',required=True)
    p.add_argument('--split',default='development',choices=['development','confirmation'])
    try:main(p.parse_args())
    except BaseException:state('failed',error=traceback.format_exc());raise
