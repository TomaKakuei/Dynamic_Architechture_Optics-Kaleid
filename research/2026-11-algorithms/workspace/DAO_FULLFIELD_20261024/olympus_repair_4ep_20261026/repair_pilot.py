from local import *
from repaired_inverse import FullCamera,refine
import traceback


@torch.no_grad()
def run():
    out=RUN/'repair_pilot';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    with np.load(SOURCE/'training/selected_validation.npz') as z:old={k:z[k].copy() for k in z.files}
    camera=FullCamera();metrics=DiffMetrics();blocks=Blocks();results=[]
    for case in (7111,7007,7049,7363):
        dest=out/f'case_{case}.json'
        if dest.exists():results.append(json.loads(dest.read_text()));continue
        slot=int(np.where(old['index']==case)[0][0]);prior=-old['command'][slot]
        images,_=blocks.read('joint',case//25)
        tick=common.clock()
        callback=lambda step,axis:state('repair_pilot',case=case,step=step,jacobian_column=axis,columns=35,completed_cases=len(results))
        estimate,history,info=refine(camera,images[case%25],prior,steps=2,callback=callback)
        # The command is fixed before labels are used for scoring.
        fields,valid=metrics.score(old['initial'][slot],-estimate)
        oldscore=json.loads((LEGACY/'evaluation500'/f'case_{case}.json').read_text())['baseline']
        row=dict(case=case,command=(-estimate).tolist(),steps=history,info=info,fields=fields[0].tolist(),
            valid=bool(valid[0]),center_pass=bool(valid[0] and fields[0,0]<.07),
            full_pass=bool(valid[0] and (fields[0]<.07).all()),baseline=oldscore,
            active_seconds=common.clock()-tick)
        common.atomic(dest,row);results.append(row);print(json.dumps(row),flush=True)
    prior_pass=[r['baseline']['valid'] and max(r['baseline']['mixed_wrms'])<.07 for r in results]
    # A local defect-removal gate only; not a generalization guarantee.
    gate=all(r['valid'] for r in results) and all(not p or r['full_pass'] for p,r in zip(prior_pass,results))
    result=dict(cases=4,gate_pass=gate,rows=results,full_pass_before=sum(prior_pass),full_pass_after=sum(r['full_pass'] for r in results),
        invalid_after=sum(not r['valid'] for r in results),training_updates=0,scope='Four known failure/control states; no independent test claim')
    common.atomic(out/'result.json',result);state('repair_pilot_completed',gate_pass=gate,cases=4)
    return result


if __name__=='__main__':
    try:run()
    except BaseException:state('repair_pilot_failed',error=traceback.format_exc());raise
