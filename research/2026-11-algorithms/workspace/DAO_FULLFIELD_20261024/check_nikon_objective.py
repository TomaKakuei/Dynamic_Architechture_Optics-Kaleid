"""Compare two physical objectives at eight frozen states; zero weight updates."""
import json
import numpy as np
import torch
from experiment import System, HERE, atomic
from nikon_objective import physical_per_sample, VERSION

out=HERE/'nikon_debug_20261024'
system=System('nikon')
with np.load(out/'small_sample_endpoints.npz') as z:
    q=torch.tensor(z['large_fullfield'][96:104],device='cuda',dtype=torch.float64)

def fields(q):return system.metrics(q,q.detach())[0]
def old(w):
    ell=torch.log1p((w/.07).square());term=ell/(1+ell)
    return (.5*term.mean(1)+.5*term.amax(1)).mean()
def new(w):return physical_per_sample(w).mean()

results={}
for label,objective in [('bounded_log_v1',old),(VERSION,new)]:
    state=q.clone().requires_grad_();loss=objective(fields(state));g=torch.autograd.grad(loss,state)[0]
    assert torch.isfinite(g).all()
    direction=g/g.norm();h=1e-4
    with torch.no_grad():fd=(objective(fields(q+h*direction))-objective(fields(q-h*direction)))/(2*h)
    analytic=(g*direction).sum()
    # One identical coordinate-sized diagnostic probe, not an optimizer/network update.
    step=.001*g/g.abs().amax(dim=1,keepdim=True)
    with torch.no_grad():before=fields(q);after=fields(q-step)
    results[label]=dict(loss=float(loss),gradient_norm=float(g.norm()),finite=True,
        finite_difference_relative_error=float(abs(fd-analytic)/abs(analytic)),
        initial_field_mean=before.mean(0).tolist(),probe_field_mean=after.mean(0).tolist(),
        probe_max_coordinate_change=.001)
model=system.model('large').eval()
model.load_state_dict(torch.load(HERE/'nikon_large/candidate.pt',map_location='cpu',weights_only=False)['model'])
with np.load(out/'small_sample_endpoints.npz') as z:
    data={k:z[k].copy() for k in ('image','raw','q')}
loss,parts=system.loss(model,data,np.arange(96,98));loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
results['production_loss_smoke']=dict(loss=float(loss),parts=parts,finite_gradients=True,
    parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),optimizer_updates=0)
atomic(out/'objective_check.json',dict(no_training=True,results=results))
print(json.dumps(results,indent=2))
