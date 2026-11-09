"""Single-packet exact-response preflight; no optimizer updates."""
from short_train import TrialSystem,HERE,exp,torch,np
import json,time
s=TrialSystem('exact_local_response');data=s.load_data();model=s.model('large').eval()
rng=np.random.default_rng(202610243)
ids=np.concatenate([rng.choice(np.flatnonzero(data['train']&(data['cohort']==g)),32,replace=False)
                    for g in np.unique(data['cohort'])])
torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
with torch.no_grad():
    endpoint,initial,command,pose=s.forward(model,data,ids)
    fields,valid,absolute=s.metrics(endpoint,initial)
    residual=s.native_optics.metrics(-command.double())['difference']-torch.as_tensor(data['raw'][ids],device='cuda')
    result=exp.aggregate(fields.cpu().numpy(),valid.cpu().numpy(),absolute.cpu().numpy(),.07)
forward_seconds=time.perf_counter()-start
model.train();start=time.perf_counter();loss,parts=s.loss(model,data,ids);loss.backward();torch.cuda.synchronize()
report=dict(count=len(ids),training_indices=ids.tolist(),score=result,
    observation_rms_waves=float(residual.square().mean().sqrt()),loss=float(loss.detach()),parts=parts,
    gradient_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
    gradient_parameter_tensors=sum(p.grad is not None for p in model.parameters()),
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,forward_seconds=forward_seconds,
    training_seconds=time.perf_counter()-start,optimizer_updates=0,
    backward='Implicit GN approximation at terminal solution; not an exact unrolled derivative')
exp.atomic(HERE/'exact_response_preflight.json',report)
assert torch.isfinite(loss) and report['gradient_finite'],report
print(json.dumps({k:v for k,v in report.items() if k!='training_indices'},indent=2))
