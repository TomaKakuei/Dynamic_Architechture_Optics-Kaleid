"""Round1: objective intervention at frozen network outputs, no network update."""
from nikon_lab import *
import json
lab=Lab();start=active_time();results={};arrays={}
def loss(w,kind):
    if kind.startswith('bounded'):
        e=torch.log1p((w/.07).square());t=e/(1+e)
    else:t=torch.sqrt(1+(w/.07).square())-1
    value=.5*t.mean(1)+.5*t.max(1).values
    return value*(100 if kind=='bounded_x100' else 1)

for kind in ('bounded','bounded_x100','pseudohuber'):
    q=torch.nn.Parameter(tensor(lab.q0).clone());opt=torch.optim.Adam([q],lr=.01)
    trace=[dict(step=0,score=lab.score(lab.q0))]
    for i in range(32):
        opt.zero_grad(set_to_none=True)
        r=lab.residual_tensor(q);w=torch.sqrt(r.square().mean(-1).clamp_min(1e-20))
        objective=loss(w,kind).mean();objective.backward()
        torch.nn.utils.clip_grad_norm_([q],5.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():q.copy_(tensor(lab.hidden)+(q-tensor(lab.hidden)).clamp(-1.5,1.5))
        if (i+1)%8==0:trace.append(dict(step=i+1,score=lab.score(q.detach().cpu().numpy())))
    arrays[kind]=q.detach().cpu().numpy();results[kind]=dict(steps=32,trace=trace)
    print(kind,brief(trace[-1]['score']),flush=True)

# Separate oracle response control: uses true full-field residual/Jacobian.
# It establishes feasibility/conditioning, NOT an online method or matched optimizer speed claim.
q=lab.q0.copy();trace=[dict(step=0,score=lab.score(q))]
for i in range(4):
    j=lab.jacobian(q,'audit');r=lab.observation(q,'audit')
    step=cap(-(inverse(j)@r[...,None])[...,0]);current=(r*r).sum(1)
    candidates=np.concatenate([lab.hidden+np.clip(q+gain*step-lab.hidden,-1.5,1.5) for gain in (1,.5,.25)])
    values=lab.observation(candidates,'audit').reshape(3,len(q),-1)
    costs=(values*values).sum(-1);idx=np.argmin(costs,axis=0)
    trial=candidates.reshape(3,len(q),35)[idx,np.arange(len(q))]
    q=np.where((costs[idx,np.arange(len(q))]<current)[:,None],trial,q)
    trace.append(dict(step=i+1,score=lab.score(q)))
arrays['oracle_gn4']=q;results['oracle_gn4']=dict(steps=4,truth_used=True,trace=trace)
results['scale_control_max_endpoint_difference']=float(abs(arrays['bounded']-arrays['bounded_x100']).max())
atomic(HERE/'round1.json',dict(round=1,status='completed',network_updates=0,scope='24old validation states, frozen outputs',
                             results=results,effective_seconds=active_time()-start))
np.savez_compressed(HERE/'round1_endpoints.npz',**arrays)
print('oracle_gn4',brief(trace[-1]['score']),flush=True)
print('scale_control_maxdiff',results['scale_control_max_endpoint_difference'],flush=True)
