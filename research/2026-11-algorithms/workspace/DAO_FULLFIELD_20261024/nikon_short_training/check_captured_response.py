from short_train import TrialSystem,HERE,exp,torch,np
import time,json
s=TrialSystem('exact_local_response');data=s.load_data();model=s.model('large').eval()
ids=np.flatnonzero(data['train'])[np.random.default_rng(202610244).choice(24000,96,replace=False)]
image=torch.as_tensor(data['image'][ids],device='cuda');raw=torch.as_tensor(data['raw'][ids],device='cuda')
with torch.no_grad():
    prediction=exp.LargerNetwork.forward(model,image,raw)
    physical=(prediction*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5).double()
    model.captured_fd_response(physical);torch.cuda.synchronize()
    start=time.perf_counter();reference=model.finite_response(physical);torch.cuda.synchronize();plain=time.perf_counter()-start
    start=time.perf_counter();captured=model.captured_fd_response(physical);torch.cuda.synchronize();replay=time.perf_counter()-start
    changed=physical+.0003
    moved=model.captured_fd_response(changed);truth=model.finite_response(changed)
    result=dict(count=len(ids),max_absolute_error=float((captured-reference).abs().max()),
        changed_input_max_absolute_error=float((moved-truth).abs().max()),plain_seconds=plain,replay_seconds=replay,
        speed_ratio=plain/replay,scope='Same96states and exact same FP64 finite-difference response; no concurrent GPU workload')
exp.atomic(HERE/'captured_response_check.json',result)
assert result['max_absolute_error']<1e-8 and result['changed_input_max_absolute_error']<1e-8,result
print(json.dumps(result,indent=2))
