from short_train import TrialSystem,HERE,exp,torch,np
from fp32_response import make_float_optics,response
import time,json
s=TrialSystem('exact_local_response');data=s.load_data();model=s.model('large').eval()
rng=np.random.default_rng(202610243)
ids=np.concatenate([rng.choice(np.flatnonzero(data['train']&(data['cohort']==g)),32,replace=False) for g in np.unique(data['cohort'])])
image=torch.as_tensor(data['image'][ids],device='cuda');raw=torch.as_tensor(data['raw'][ids],device='cuda')
o32=make_float_optics(s.native_optics)
with torch.no_grad():
    prediction=exp.LargerNetwork.forward(model,image,raw)
    p=(prediction*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5).double()
    j=model.finite_response(p);j32=response(o32,p)
    start=time.perf_counter();j64ad=response(s.native_optics,p);torch.cuda.synchronize();ad64_seconds=time.perf_counter()-start
    residual=model.difference(p)-raw.flatten(1).double()/7.
    def step(jac):
        jt=jac.transpose(1,2);g=jt@jac+1e-4*torch.eye(35,device='cuda',dtype=torch.float64)
        d=-torch.linalg.solve(g,(jt@residual.unsqueeze(-1))).squeeze(-1)
        return d/torch.clamp(d.abs().amax(1,keepdim=True)/.25,min=1.)
    d=step(j);d32=step(j32)
    delta=float((d-d32).norm()/d.norm())
    model.response=lambda p:response(o32,p)
    start=time.perf_counter();endpoint,initial,command,pose=s.forward(model,data,ids);torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    fields,valid,absolute=s.metrics(endpoint,initial)
    score=exp.aggregate(fields.cpu().numpy(),valid.cpu().numpy(),absolute.cpu().numpy(),.07)
report=dict(count=len(ids),fp32_step_relative_difference=delta,fp64_ad_relative_jacobian_error=float((j64ad-j).norm()/j.norm()),
    fp64_ad_seconds=ad64_seconds,fp32_full_inference_seconds=seconds,fp32_score=score,
    reference_fp64_passes=21,scope='Same96training cases; numeric-backend decision before actual4epoch trial')
exp.atomic(HERE/'response_effect_check.json',report)
print(json.dumps(report,indent=2))
