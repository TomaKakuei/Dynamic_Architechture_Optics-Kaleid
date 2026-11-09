"""Check the new batched loop against every original11case and derivatives."""
from short_train import TrialSystem,HERE,exp,torch,np
from nikon_closed_loop import NikonFeedback,run
from feedback_training import MeasuredFeedback
import json,time

adapter=NikonFeedback();feedback=MeasuredFeedback(adapter.system)
from fast_execution import nikon_execution
seeds=list(adapter.nk.TEST_SEEDS);initial=[];proposals=[];raw=[];reference=[]
with nikon_execution():
    for seed in seeds:
        hidden=adapter.nk.sample(seed)[0]
        packet=adapter.nk.Plant(hidden,noise_seed=seed,noise_std=.015,score_center=True).measure(np.zeros(35))
        initial.append(hidden);proposals.append(-np.clip(adapter.predict(packet),-1.5,1.5));raw.append(packet['network'])
        reference.append(run(adapter,seed,'diff_calibrated'))
q=torch.tensor(np.array(initial),dtype=torch.float64,device='cuda')
p=torch.tensor(np.array(proposals),dtype=torch.float64,device='cuda')
r=torch.tensor(np.array(raw),dtype=torch.float64,device='cuda')
with torch.no_grad():
    command=feedback.run(q,p,seeds,r)
    fields,valid,_=adapter.system.metrics(q+command,q)
    center,center_valid=feedback.center_signal(q)
    old_center=adapter.system.optics.metrics(q)['absolute'][:,0]
    center_error=float((center-old_center).abs().max())
    counts=feedback.last_counts.copy()
commands=np.array([v['command'] for v in reference])
old_fields=np.array([[v['terminal']['center_abs']]+v['terminal']['edge_diff'] for v in reference])
command_error=float(np.max(np.abs(command.cpu().numpy()-commands)))
field_error=float(np.max(np.abs(fields.cpu().numpy()-old_fields)))
old_counts=np.array([v['counts']['acquisitions'] for v in reference])
parity=dict(count=11,command_max_abs=command_error,field_max_abs=field_error,
    center_trace_max_abs=center_error,counts=counts.tolist(),reference_counts=old_counts.tolist(),
    fullfield_passes=int((fields.amax(1)<.07).sum()),reference_passes=sum(v['terminal']['fullfield_pass'] for v in reference))
exp.atomic(HERE/'feedback_parity.json',parity)
assert command_error<1e-5 and field_error<1e-5 and center_error<1e-8,parity
assert np.array_equal(counts,old_counts),parity
print('PARITY',json.dumps(parity),flush=True)

def objective(proposal):
    final=feedback.run(q,proposal,seeds,r)
    w,_,_=adapter.system.metrics(q+final,q)
    ell=torch.log1p((w/.07).square());term=ell/(1+ell)
    return (.5*term.mean(1)+.5*term.amax(1)).mean()
p=p.detach().requires_grad_(True)
value=objective(p);g=torch.autograd.grad(value,p)[0]
v=torch.tensor(np.random.default_rng(4).normal(size=p.shape),device='cuda');v=v/v.norm()
analytic=float((g*v).sum());checks=[]
with torch.no_grad():
    for step in (1e-4,3e-5):
        plus=objective(p+step*v);pc=feedback.last_counts.copy()
        minus=objective(p-step*v);mc=feedback.last_counts.copy()
        difference=float((plus-minus)/(2*step))
        checks.append(dict(step=step,analytic=analytic,finite_difference=difference,
            absolute_error=abs(difference-analytic),relative_error=abs(difference-analytic)/max(abs(analytic),1e-8),
            same_read_counts=bool(np.array_equal(pc,counts) and np.array_equal(mc,counts))))
exp.atomic(HERE/'feedback_gradient_check.json',dict(checks=checks))
assert all(v['same_read_counts'] and (v['relative_error']<.01 or v['absolute_error']<1e-5) for v in checks),checks
print('GRADIENT',json.dumps(checks),flush=True)

system=TrialSystem('closed_loop_training');data=system.load_data();model=system.model('large').train()
ids=np.random.default_rng(202610241).choice(np.flatnonzero(data['train']),128,replace=False)
torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
loss,parts=system.loss(model,data,ids);loss.backward();torch.cuda.synchronize()
assert torch.isfinite(loss) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
smoke=dict(batch=128,loss=float(loss.detach()),parts=parts,seconds=time.perf_counter()-start,
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,optimizer_updates=0,
    gradient_parameter_tensors=sum(p.grad is not None for p in model.parameters()),
    all_network_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
exp.atomic(HERE/'feedback_training_preflight.json',smoke)
print('TRAINING_PREFLIGHT',json.dumps(smoke),flush=True)
