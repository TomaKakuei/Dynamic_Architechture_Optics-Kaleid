"""Local model refinement interface/finite-gradient smoke; no optimizer step."""
from short_train import TrialSystem,HERE,exp,torch,np
import time,json
s=TrialSystem('nonlinear_local_model');data=s.load_data();model=s.model('large').eval()
rng=np.random.default_rng(202610243)
ids=np.concatenate([rng.choice(np.flatnonzero(data['train']&(data['cohort']==g)),32,replace=False)
                    for g in np.unique(data['cohort'])])
torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
with torch.no_grad():
    endpoint,initial,command,pose=s.forward(model,data,ids)
    w,valid,absolute=s.metrics(endpoint,initial)
    innovation=s.native_optics.metrics(-command.double())['difference']-torch.as_tensor(data['raw'][ids],device='cuda')
    score=exp.aggregate(w.cpu().numpy(),valid.cpu().numpy(),absolute.cpu().numpy(),.07)
model.train();loss,parts=s.loss(model,data,ids);loss.backward();torch.cuda.synchronize()
report=dict(count=len(ids),training_indices=ids.tolist(),score=score,
    observation_rms_waves=float(innovation.square().mean().sqrt()),loss=float(loss.detach()),parts=parts,
    gradient_finite=all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),
    gradient_parameter_tensors=sum(p.grad is not None for p in model.parameters()),
    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,seconds=time.perf_counter()-start,
    optimizer_updates=0,measurements_per_inference=1,model_only_refinements=4)
exp.atomic(HERE/'local_model_preflight.json',report)
assert torch.isfinite(loss) and report['gradient_finite'],report
print(json.dumps(report,indent=2))
