"""Same selected weights, one neural forward only; no inverse-model steps."""
from short_train import TrialSystem,HERE,exp,torch,np

@torch.no_grad()
def predict(model,images,raw):
    prediction=model(torch.as_tensor(images,dtype=torch.float32,device='cuda'),
        torch.as_tensor(raw,dtype=torch.float32,device='cuda'))
    return -(prediction*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5)

@torch.no_grad()
def main():
    folder=HERE/'exact_local_response'
    system=TrialSystem('bounded_control')
    model=system.model('large').eval()
    assert type(model) is exp.LargerNetwork
    saved=torch.load(folder/'candidate.pt',map_location='cpu',weights_only=False)
    original_keys=set(model.state_dict())
    removed=sorted(set(saved['model'])-original_keys)
    assert removed and all(k.startswith('native_optics.') for k in removed)
    model.load_state_dict({k:saved['model'][k] for k in original_keys},strict=True)
    assert all(torch.equal(v.cpu(),saved['model'][k]) for k,v in model.state_dict().items())
    data=system.load_data();indices=np.flatnonzero(~data['train'])
    commands=[]
    # Pose labels and the optical model are absent from this prediction call.
    for start in range(0,len(indices),128):
        ids=indices[start:start+128]
        commands.append(predict(model,data['image'][ids],data['raw'][ids]).cpu().numpy())
    command=np.concatenate(commands)
    assert command.shape==(6000,35) and np.isfinite(command).all()
    fields=[];valids=[];absolutes=[]
    # Only after every command is frozen, use hidden poses for scoring.
    for start in range(0,len(indices),128):
        ids=indices[start:start+128]
        q=torch.as_tensor(data['q'][ids],dtype=torch.float64,device='cuda')
        u=torch.as_tensor(command[start:start+128],dtype=torch.float64,device='cuda')
        f,v,a=system.metrics(q+u,q)
        fields.append(f.cpu().numpy());valids.append(v.cpu().numpy());absolutes.append(a.cpu().numpy())
    f,v,a=[np.concatenate(items) for items in (fields,valids,absolutes)]
    metrics=exp.aggregate(f,v,a,.07)
    cohorts=data['cohort'][indices]
    metrics['cohorts']={str(g):exp.aggregate(f[cohorts==g],v[cohorts==g],a[cohorts==g],.07) for g in np.unique(cohorts)}
    np.savez_compressed(folder/'pure_network_validation.npz',indices=indices,original_ids=data['ids'][indices],
        cohort=cohorts,initial=data['q'][indices],command=command,fields=f,valid=v,absolute=a)
    result=dict(updated=exp.stamp(),checkpoint=str(folder/'candidate.pt'),epoch=int(saved['epoch']),
        protocol='One measurement, one neural forward, fixed output de-normalization/clipping, no model refinement, no feedback',
        prediction_inputs=['raw','maps'],truth_used_for_prediction=False,optical_model_used_for_prediction=False,
        same_neural_parameters_and_buffers=True,network_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        trainable_parameter_count=sum(p.numel() for p in model.parameters()),network_training=False,
        model_refinements=0,measurement_acquisitions=1,network_forwards_per_case=1,
        validation=metrics,scope='Same6000original selection poses and original noise; not an independent test')
    exp.atomic(folder/'pure_network_summary.json',result)
    print(__import__('json').dumps(result,indent=2))

if __name__=='__main__':main()
