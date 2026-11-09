"""Original-resolution image inverse with complete finite response and prior trust.

The full-response difference includes phase, carrier, pupil boundary, pixel
integration and normalization. This bounded diagnostic path is intentionally
separate from the efficient cached-image network training loop.
"""
from local import *


class FullCamera(CameraModel):
    def __init__(self):super().__init__(n=2048,degree=24)

    @torch.no_grad()
    def evaluate(self,q,derivative=False,callback=None):
        if derivative:return self.response(q,callback)
        rows=self.representations(q,False)
        if max(self.last_diagnostics['phase_fit_rms'])>.0003:
            raise ValueError('Phase representation outside original fit tolerance; retain prior')
        images=[self.field(row,False)[0] for row in rows]
        return torch.stack(images),None

    @torch.no_grad()
    def response(self,q,callback=None,h=.01):
        central,_=self.evaluate(q);columns=[]
        for axis in range(35):
            plus=np.array(q,copy=True);minus=np.array(q,copy=True)
            plus[axis]=min(1.,plus[axis]+h);minus[axis]=max(-1.,minus[axis]-h)
            a,_=self.evaluate(plus);b,_=self.evaluate(minus)
            columns.append((a-b)/(plus[axis]-minus[axis]))
            if callback:callback(axis+1)
        return central,torch.stack(columns,dim=-1)


@torch.no_grad()
def refine(camera,image,prior,steps=2,callback=None):
    """No truth or validation score is accepted as input."""
    prior=np.asarray(prior,dtype=np.float64);x=prior.copy()
    image=torch.as_tensor(image,device='cuda').float();sigma=sigma_from_adu(image)
    identity=torch.eye(35,device='cuda',dtype=torch.float64)
    base_ridge=.01;radius=.1;step_radius=.05;history=[]
    for step in range(steps):
        try:
            prediction,j=camera.evaluate(x,True,callback=(lambda axis:callback(step+1,axis)) if callback else None)
        except (ValueError,RuntimeError) as exc:
            return x,history,{'reason':'model_failure_retained_last','error':str(exc)}
        r=((prediction-image)/sigma/128).flatten().double()
        J=(j/sigma[...,None]/128).reshape(-1,35).double()
        departure=torch.as_tensor(x-prior,device='cuda').double()
        gram=J.T@J;gradient=J.T@r+base_ridge*departure
        damping=base_ridge
        # Increase curvature damping, rather than scale every direction by an
        # uncontrolled weak-coordinate proposal. Total departure is also bounded.
        for _ in range(30):
            delta=-torch.linalg.solve(gram+damping*identity,gradient)
            if float(delta.abs().max())<=step_radius:break
            damping*=2
        delta=delta.cpu().numpy()
        oldcost=float(r.square().sum()+base_ridge*departure.square().sum());newcost=oldcost;accepted=False;gain_used=0
        for gain in (1.,.5,.25):
            trial=np.clip(x+gain*delta,np.maximum(-1,prior-radius),np.minimum(1,prior+radius))
            try:
                if not camera.camera.engine.geometry.path(trial)['safe']:continue
                # A prior-based estimated correction-path check; it is NOT a
                # certificate for an unknown actual initial state.
                if not camera.camera.engine.geometry.path(prior-trial,start=prior)['safe']:continue
                im,_=camera.evaluate(trial)
                merit=float((((im-image)/sigma/128).square().sum())+base_ridge*np.sum((trial-prior)**2))
            except (ValueError,RuntimeError):continue
            if np.isfinite(merit) and merit<oldcost:
                x=trial;newcost=merit;accepted=True;gain_used=gain;break
        history.append(dict(step=step+1,before=oldcost,after=newcost,damping=damping,gain=gain_used,
                            max_departure=float(abs(x-prior).max()),accepted=accepted))
        if not accepted:break
    return x,history,dict(reason='bounded_steps_complete',total_departure_limit=radius,step_limit=step_radius)
