"""Training-only linear readout initialization; no extra optimizer epochs."""
from repair import *
import hashlib,os

def main():
    assert not (HERE/'initialized.pt').exists(), 'Preserve the registered initialization'
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    system=RepairSystem();data=system.load_data();indices=np.flatnonzero(data['train'])
    model=system.fresh_model().eval()
    rng=np.random.default_rng(SEED)
    probe=rng.choice(indices,128,replace=False)
    before=exp.evaluate(system,model,data,probe,128,HERE/'pre_repair_training_probe.npz')
    features=[]
    tick=exp.active_time()
    with torch.no_grad():
        for start in range(0,len(indices),256):
            ix=indices[start:start+256]
            image=torch.as_tensor(data['image'][ix],device='cuda')
            raw=torch.as_tensor(data['raw'][ix],device='cuda')
            features.append(model.features(image,raw).cpu())
    x=torch.cat(features)
    y=(torch.as_tensor(data['q'][indices],dtype=torch.float32)-model.base.target_mean.cpu())/model.base.target_std.cpu()
    # Standardized ridge regression. Float64 is a small CPU initialization solve;
    # every deployed neural parameter and every neural forward stays float32.
    xd=x.double();yd=y.double();xm=xd.mean(0);ym=yd.mean(0)
    xs=xd.std(0).clamp_min(1e-6);z=(xd-xm)/xs
    ridge=.001
    gram=z.T@z/len(z)+ridge*torch.eye(z.shape[1],dtype=torch.float64)
    beta=torch.linalg.solve(gram,z.T@(yd-ym)/len(z))
    weight=(beta/xs[:,None]).T;bias=ym-xm@(beta/xs[:,None])
    with torch.no_grad():
        model.head[-1].weight.copy_(weight.float().cuda())
        model.head[-1].bias.copy_(bias.float().cuda())
    prediction=x@model.head[-1].weight.cpu().T+model.head[-1].bias.cpu()
    fit_mse=float((prediction-y).square().mean().detach())
    exp.save_pt(HERE/'initialized.pt',dict(model=model.state_dict(),config=system.trial_config,
        epoch=0,initialization_fit_samples=len(indices),optimizer_updates=0))
    after=exp.evaluate(system,model,data,probe,128,HERE/'post_repair_training_probe.npz')
    model.train();model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();step_tick=exp.active_time()
    loss,parts=system.loss(model,data,probe);loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    torch.cuda.synchronize();step_seconds=exp.active_time()-step_tick
    parameters=list(model.named_parameters())
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in parameters)
    assert len(parameters)==101
    identity=model.base.target_mean.detach().cpu().numpy()
    # Direct prior gradient reaches the full head before the approximate implicit
    # optical derivative. The archived head is not re-used as a complete output.
    evidence=dict(passed=True,updated=exp.stamp(),pid=os.getpid(),
        parameters=PARAMETERS,parameter_tensors=len(parameters),network_dtype='float32',
        grouped_subnet_parameters=0,optimizer_probe_updates=0,
        initialization_fit_samples=len(indices),fit_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        validation_samples_used_for_initialization=0,readout_fit_is_training=True,
        initialization_ridge=ridge,normalized_pose_fit_mse=fit_mse,
        training_probe_indices=probe.tolist(),training_probe_is_in_sample=True,
        before=before,after=after,loss=float(loss.detach()),loss_parts=parts,
        all_gradients_finite=True,forward_backward_seconds=step_seconds,
        peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,
        initialization_and_probe_active_seconds=exp.active_time()-tick,
        estimated_four_epoch_seconds=4*(188*step_seconds+70)+240,
        inference_requires_truth=False,measured_feedback=False,
        scope='Train-only readout fit and 128 in-sample diagnostic cases; no recovery claim until full validation')
    exp.atomic(HERE/'preflight.json',evidence)
    print(json.dumps(evidence,indent=2),flush=True)

if __name__=='__main__':main()
