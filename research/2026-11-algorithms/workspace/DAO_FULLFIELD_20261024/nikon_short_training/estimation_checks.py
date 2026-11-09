"""Explain the completed short trials using their saved validation commands.

This is post-training diagnosis on the existing selection set, not a new test
set or a substitute for the required four actual epochs per idea.
"""
from short_train import TrialSystem, HERE, exp, torch, np
import json


def training_gradient_probe(system,data):
    rng=np.random.default_rng(202610242)
    ids=np.concatenate([rng.choice(np.flatnonzero(data['train'] & (data['cohort']==group)),32,replace=False)
                        for group in np.unique(data['cohort'])])
    model=system.model('large').eval()
    with torch.no_grad():
        image=torch.as_tensor(data['image'][ids],device='cuda')
        raw=torch.as_tensor(data['raw'][ids],device='cuda')
        prediction=model(image,raw)
    with torch.enable_grad():
        p=prediction.detach().requires_grad_(True)
        q=torch.as_tensor(data['q'][ids],device='cuda')
        command=-(p*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5)
        fields,valid,_=system.metrics(q+command.double(),q)
        assert bool(valid.all())
        ell=torch.log1p((fields/.07).square());term=ell/(1+ell)
        physical=(.5*term.mean(1)+.5*term.amax(1)).mean()
        pose=torch.nn.functional.smooth_l1_loss(p,((q-model.base.target_mean)/model.base.target_std).float(),beta=.25)
        gp=torch.autograd.grad(physical,p,retain_graph=True)[0]
        gq=torch.autograd.grad(pose,p)[0]
    ratio=gp.norm()/gq.norm()
    result=dict(training_cases=len(ids),training_indices=ids.tolist(),network_mode='eval; no updates',
        comparison_space='normalized network output35coordinates',physical_loss=float(physical.detach()),
        pose_loss=float(pose.detach()),physical_gradient_norm=float(gp.norm()),pose_gradient_norm=float(gq.norm()),
        physical_over_unweighted_pose_gradient=float(ratio),
        physical_over_original_weighted_pose_gradient=float(ratio/.02),
        physical_pose_gradient_cosine=float((gp*gq).sum()/(gp.norm()*gq.norm())),
        scope='Output-gradient diagnosis on96original training cases; no network optimizer updates')
    exp.atomic(HERE/'training_gradient_probe.json',result)
    print('training_gradient_probe',json.dumps({k:v for k,v in result.items() if k!='training_indices'}),flush=True)


@torch.no_grad()
def main():
    system=TrialSystem('sensor_consistency')
    data=system.load_data()
    validation=np.flatnonzero(~data['train'])
    raw=torch.as_tensor(data['raw'][validation],device='cuda')
    targets=data['q'][validation]
    native=system.native_optics
    def sensor_error(pose):
        residuals=[]
        invalid=0
        for start in range(0,len(pose),128):
            prediction=torch.as_tensor(pose[start:start+128],dtype=torch.float64,device='cuda')
            metrics=native.metrics(prediction)
            residuals.append((metrics['difference']-raw[start:start+128]).cpu().numpy())
            invalid+=int((~metrics['valid']).sum())
        residual=np.concatenate(residuals)
        return dict(pointwise_rms_waves=float(np.sqrt(np.mean(residual**2))),
            field_rms_waves=np.sqrt(np.mean(residual**2,axis=(0,2))).tolist(),invalid=invalid)
    results={'truth_noise_reference':sensor_error(targets)}
    jobs=json.loads((HERE/'plan.json').read_text())['jobs']
    sources=[('original_epoch49',HERE/'bounded_control/baseline_validation.npz')]
    sources += [(idea,HERE/idea/('one_shot_validation_004.npz' if idea=='closed_loop_training' else 'validation_004.npz')) for idea in jobs
                if (HERE/idea/'training_evidence.json').exists()]
    for name,path in sources:
        with np.load(path) as z:
            assert np.array_equal(z['indices'],validation)
            assert np.array_equal(z['initial'],targets)
            predicted=-z['command']
            error=predicted-targets
            result=dict(pose_coordinate_rms=float(np.sqrt(np.mean(error**2))),
                pose_sample_l2_mean=float(np.linalg.norm(error,axis=1).mean()),
                clamped_coordinate_fraction=float((np.abs(predicted)>=1.5).mean()),
                observation=sensor_error(predicted),source=str(path))
        results[name]=result
        print(name,json.dumps(result),flush=True)
    exp.atomic(HERE/'estimation_checks.json',dict(updated=exp.stamp(),count=len(validation),
        scope='Existing6000selection cases; original native0,+/-0.05mm noisy raw DIFF inputs',
        diagnosis_only=True,results=results))
    training_gradient_probe(system,data)


if __name__=='__main__':
    main()
