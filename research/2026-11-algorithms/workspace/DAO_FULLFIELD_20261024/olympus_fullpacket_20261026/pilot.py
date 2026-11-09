"""Frozen Olympus candidate, eight fixed original validation packets.
Original center criterion retained; nine-field mean and worst separately shown.
"""
from support import *
from camera_inverse import CameraModel
import json,traceback,os


def score(obj,q,estimate):
    endpoint=torch.as_tensor((q-estimate)[None],device='cuda').double()
    initial=torch.as_tensor(q[None],device='cuda').double()
    safe=bool(obj.path_valid(endpoint,initial)[0])
    with torch.no_grad():
        var,_,pupil,chief=obj.full.high_order(endpoint)
        fields=var[0].clamp_min(0).sqrt().cpu().numpy()
        valid=safe and bool(chief.all()) and bool((pupil.double().mean(-1)>=.2).all()) and bool(np.isfinite(fields).all())
    return dict(fields=fields.tolist(),valid=valid,center_pass=bool(valid and fields[0]<.07),
                mean08_center08=bool(valid and fields.mean()<.08 and fields[0]<.08),
                all9_08=bool(valid and fields.max()<.08))


def main():
    here=HERE/'pilot';here.mkdir(exist_ok=True)
    if (here/'summary.json').exists():print((here/'summary.json').read_text());return
    with np.load(SOURCE/'training/selected_validation.npz') as z:old={k:z[k].copy() for k in z.files}
    selected=np.random.default_rng(2026102607).choice(500,8,replace=False)
    common.atomic(here/'selection.json',dict(seed=2026102607,indices=old['index'][selected].tolist(),selected_before_new_inference=True))
    state('initializing_camera',pid=os.getpid(),pilot_cases=8)
    camera=CameraModel();obj=Objective();blocks=Blocks();results=[];total_tick=common.clock()
    for slot in selected:
        case=int(old['index'][slot]);dest=here/f'case_{case}.json'
        if dest.exists():results.append(json.loads(dest.read_text()));continue
        images,targets=blocks.read('joint',case//25);image=torch.as_tensor(images[case%25],device='cuda').float()
        # Saved frozen neural prediction is reused; only the noisy image enters refinement.
        prior=-old['command'][slot];x=prior.copy();sigma=sigma_from_adu(image);scale=128.
        steps=[];tick=common.clock();best=None
        for step in range(8):
            state('pilot',case=case,completed_cases=len(results),step=step+1,steps=8,pid=os.getpid())
            expected,j=camera.evaluate(x,derivative=True)
            residual=((expected-image)/sigma/scale).flatten().double()
            jj=(j/sigma[...,None]/scale).reshape(-1,35).double()
            gram=jj.T@jj+1e-4*torch.eye(35,device='cuda',dtype=torch.float64)
            dx=torch.as_tensor(x-prior,device='cuda').double()
            delta=-torch.linalg.solve(gram,jj.T@residual+1e-4*dx)
            delta=delta/torch.clamp(delta.abs().max()/.25,min=1.)
            objective=float(residual.square().sum()+1e-4*dx.square().sum())
            accepted=False;chosen=0.;new_objective=objective
            for gain in (1.,.5,.25):
                trial=np.clip(x+gain*delta.cpu().numpy(),-1,1)
                try:
                    if not camera.camera.engine.geometry.path(trial)['safe']:continue
                    pred,_=camera.evaluate(trial)
                    merit=float((((pred-image)/sigma/scale).square().sum())+1e-4*np.sum((trial-prior)**2))
                except (ValueError,RuntimeError):continue
                if np.isfinite(merit) and merit<objective:
                    x=trial;new_objective=merit;chosen=gain;accepted=True;break
            steps.append(dict(step=step+1,objective_before=objective,objective_after=new_objective,gain=chosen,
                              accepted=accepted,model_diagnostics=camera.last_diagnostics))
            common.atomic(here/f'partial_{case}.json',dict(case=case,estimate=x.tolist(),steps=steps))
            if step==3:at4=score(obj,old['initial'][slot],x)
        terminal=score(obj,old['initial'][slot],x)
        bf=old['wrms'][slot];bv=bool(old['valid'][slot])
        row=dict(case=case,baseline=dict(fields=bf.tolist(),valid=bv,center_pass=bool(bv and bf[0]<.07),
            mean08_center08=bool(bv and bf.mean()<.08 and bf[0]<.08),all9_08=bool(bv and bf.max()<.08)),
            after4=at4,after8=terminal,command=(-x).tolist(),steps=steps,active_seconds=common.clock()-tick,
            observation_packets=1,training_updates=0,hidden_pose_used_only_for_scoring=True)
        common.atomic(dest,row);results.append(row)
        print(json.dumps(dict(case=case,baseline=row['baseline'],after4=at4,after8=terminal,seconds=row['active_seconds'])),flush=True)
    summary=dict(cases=8,rows=results,active_seconds=common.clock()-total_tick,
                 acquired_new_packets=0,training_updates=0,independent_review=False)
    common.atomic(here/'summary.json',summary);state('pilot_completed',cases=8,active_seconds=summary['active_seconds'])


if __name__=='__main__':
    try:main()
    except BaseException:state('failed',error=traceback.format_exc());raise
