from short_train import TrialSystem,HERE,exp,torch,np
from fp32_response import make_float_optics,response
import time,json
s=TrialSystem('exact_local_response');data=s.load_data();model=s.model('large').eval()
ids=np.flatnonzero(data['train'])[np.random.default_rng(202610244).choice(24000,96,replace=False)]
image=torch.as_tensor(data['image'][ids],device='cuda');raw=torch.as_tensor(data['raw'][ids],device='cuda')
o32=make_float_optics(s.native_optics)
with torch.no_grad():
    prediction=exp.LargerNetwork.forward(model,image,raw)
    physical=(prediction*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5).double()
    response(o32,physical[:1]);torch.cuda.synchronize()
    start=time.perf_counter();j64=model.response(physical);torch.cuda.synchronize();fd_seconds=time.perf_counter()-start
    start=time.perf_counter();j32=response(o32,physical);torch.cuda.synchronize();ad_seconds=time.perf_counter()-start
    error=j32-j64
    result=dict(count=len(ids),max_absolute_error=float(error.abs().max()),
        frobenius_relative_error=float(error.norm()/j64.norm()),
        sample_relative_error_max=float((error.flatten(1).norm(dim=1)/j64.flatten(1).norm(dim=1)).max()),
        fd_seconds=fd_seconds,ad_seconds=ad_seconds,finite=bool(torch.isfinite(j32).all()),
        use='FP32 derivatives only; FP64 nonlinear residual and terminal scoring unchanged')
exp.atomic(HERE/'fp32_response_check.json',result)
print(json.dumps(result,indent=2))
assert result['finite'] and result['sample_relative_error_max']<.01,result
