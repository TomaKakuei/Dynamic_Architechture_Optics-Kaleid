"""Input-only refinement: one saved camera packet and frozen neural prior.

No initial truth or scoring function is accepted by this entry point.
"""
from support import *
import json


@torch.no_grad()
def refine(camera,image,prior,checkpoint=None,callback=None):
    prior=np.asarray(prior,dtype=np.float64)
    saved=json.loads(checkpoint.read_text()) if checkpoint and checkpoint.exists() else {}
    x=np.array(saved.get('estimate',prior));steps=saved.get('steps',[])
    active=float(saved.get('active_seconds',0));tick=common.clock()
    stop=saved.get('stop_reason');image=torch.as_tensor(image,device='cuda').float()
    sigma=sigma_from_adu(image);eye=torch.eye(35,device='cuda',dtype=torch.float64)
    def snapshot():
        payload=dict(estimate=x.tolist(),steps=steps,stop_reason=stop,
                     active_seconds=active+common.clock()-tick)
        if checkpoint:common.atomic(checkpoint,payload)
        return payload
    if stop:return saved
    for step in range(len(steps),8):
        if callback:callback(step+1)
        try:
            expected,j=camera.evaluate(x,derivative=True)
            residual=((expected-image)/sigma/128.).flatten().double()
            jj=(j/sigma[...,None]/128.).reshape(-1,35).double()
            dx=torch.as_tensor(x-prior,device='cuda').double()
            delta=-torch.linalg.solve(jj.T@jj+1e-4*eye,jj.T@residual+1e-4*dx)
            delta=delta/torch.clamp(delta.abs().max()/.25,min=1.)
            objective=float(residual.square().sum()+1e-4*dx.square().sum())
            if not bool(torch.isfinite(delta).all()) or not np.isfinite(objective):
                stop='nonfinite_model_step';snapshot();break
        except (ValueError,RuntimeError) as error:
            # Model-only failure handling: retain the last available estimate.
            stop='model_failure: '+str(error);snapshot();break
        chosen=0.;new_objective=objective;accepted=False
        for gain in (1.,.5,.25):
            trial=np.clip(x+gain*delta.cpu().numpy(),-1,1)
            try:
                if not camera.camera.engine.geometry.path(trial)['safe']:continue
                prediction,_=camera.evaluate(trial)
                merit=float((((prediction-image)/sigma/128.).square().sum())+1e-4*np.sum((trial-prior)**2))
            except (ValueError,RuntimeError):continue
            if np.isfinite(merit) and merit<objective:
                x=trial;new_objective=merit;chosen=gain;accepted=True;break
        steps.append(dict(step=step+1,objective_before=objective,objective_after=new_objective,
                          accepted=accepted,gain=chosen,model_diagnostics=camera.last_diagnostics))
        if not accepted:stop='no_accepted_model_step'
        if step==7:stop=stop or 'eight_step_budget'
        snapshot()
        if stop:break
    return snapshot()
