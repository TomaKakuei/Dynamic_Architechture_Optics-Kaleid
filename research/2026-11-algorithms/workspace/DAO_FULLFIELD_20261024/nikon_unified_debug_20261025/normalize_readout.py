"""Preserve the fitted full-pose map while keeping normalization explicit."""
from diagnose import *
from stable import NormalizedExact,NormalizedTrunk

def main():
    assert (HERE/'head_diagnostic_completed.json').exists()
    system=RepairSystem();data=system.load_data();train=np.flatnonzero(data['train'])
    order=np.random.default_rng(SEED+1).permutation(train);batch=order[:128];check=order[128:256]
    old=system.model('large').eval();features=[]
    with torch.no_grad():
        for start in range(0,len(train),256):
            ix=train[start:start+256]
            features.append(old.features(torch.as_tensor(data['image'][ix],device='cuda'),
                torch.as_tensor(data['raw'][ix],device='cuda')).cpu())
    x=torch.cat(features).double()
    y=(torch.as_tensor(data['q'][train],dtype=torch.float64)-old.base.target_mean.cpu())/old.base.target_std.cpu()
    xm=x.mean(0);ym=y.mean(0);xs=x.std(0).clamp_min(1e-6);z=(x-xm)/xs
    beta=torch.linalg.solve(z.T@z/len(z)+.001*torch.eye(512,dtype=torch.float64),z.T@(y-ym)/len(z))
    weights={k:v.detach().cpu().clone() for k,v in old.state_dict().items()}
    weights.update({'feature_center':xm.float(),'feature_scale':xs.float(),
        'head.3.weight':beta.T.float(),'head.3.bias':ym.float()})
    model=NormalizedExact(1648,system.source_weights).cuda();model.native_optics=system.native_optics
    model.load_state_dict(weights);model.eval()
    with torch.no_grad():
        image=torch.as_tensor(data['image'][check],device='cuda');raw=torch.as_tensor(data['raw'][check],device='cuda')
        a=FullPoseTrunk.forward(old,image,raw);b=NormalizedTrunk.forward(model,image,raw)
        difference=dict(max=float((a-b).abs().max()),rms=float((a-b).square().mean().sqrt()))
    fit=dict(readout_weight_norm=float(weights['head.3.weight'].norm()),readout_bias_norm=float(weights['head.3.bias'].norm()),
        previous_weight_norm=float(old.head[-1].weight.norm()),previous_bias_norm=float(old.head[-1].bias.norm()),
        normalized_prior_difference=difference,fit_samples=24000,validation_labels_used=0,
        normalization='Explicit fixed train-feature centering and std in FP32; identical ridge problem',
        conditioning_limit='Backbone remains sensitive to feature changes; learning rates are verified with actual optimizer steps')
    exp.save_pt(HERE/'normalized_initialized.pt',dict(model=weights,fit=fit,epoch=0))
    # The generic diagnostic probe calls FullPoseTrunk.forward. Override it with
    # the corresponding normalized neural-only predictor for this representation.
    import diagnose
    original_forward=FullPoseTrunk.forward
    FullPoseTrunk.forward=NormalizedTrunk.forward
    report=dict(fit=fit,probes={},scope='Training-only representation and actual-update checks')
    try:
        for head_lr in (1e-7,1e-5,1e-4):
            model.load_state_dict(weights)
            label='normalized_backbone1e-7_readout'+str(head_lr)
            report['probes'][label]=first_step(system,model,data,batch,check,1e-7,label,head_lr=head_lr)
            exp.atomic(HERE/'normalization_diagnosis.json',report)
            print(json.dumps(dict(label=label,fit=fit,result=report['probes'][label])),flush=True)
    finally:FullPoseTrunk.forward=original_forward
    exp.atomic(HERE/'normalization_diagnostic_completed.json',dict(updated=exp.stamp(),discarded_optimizer_updates=3))

if __name__=='__main__':main()
