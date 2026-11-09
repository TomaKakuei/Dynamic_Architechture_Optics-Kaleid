"""One bounded diagnosis:100 matched train and100 validation seven-body cases.

Frozen additive E4. Reuse validation final commands. No optimizer updates,
new acquisitions, model changes, or independent replay of old results.
"""
from pathlib import Path
import sys,json,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'nikon_additive_4ep_20261025'))
import run_trial as run
np,torch=run.np,run.torch


def describe(values):
    a=np.asarray(values)
    return dict(mean=float(a.mean()),median=float(np.median(a)),
                p90=float(np.quantile(a,.9)),max=float(a.max()))


def center_wave(optics,q):
    t=optics.trace(q.double()+optics.bias)
    hit=t['hit'][:,:,1:];direction=t['direction'][:,:,1:]
    direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
    delta=hit-t['hit'][:,:,:1];projected=(delta*direction).sum(-1)
    rad=projected.square()-delta.square().sum(-1)+100.
    sphere=t['opl'][:,:,1:]-projected-torch.sqrt(rad.clamp_min(1e-30))
    return (sphere@optics.projector.T/optics.wave).flatten(1)


def main():
    if (HERE/'result.json').exists():
        print((HERE/'result.json').read_text());return
    tick=run.exp.active_time()
    run.exp.atomic(HERE/'state.json',dict(status='loading'))
    system=run.old.make_system('nikon');system.__class__=run.AdditiveSystem
    data=run.old.load_data(system)
    data['extra']=np.load(run.HERE/'data/extra.npy',mmap_mode='r')
    q=data['q'];ids=data['ids'];rms=np.sqrt((q*q).mean(1))
    full=(np.max(np.abs(q.reshape(-1,7,5)),axis=2)>1e-10).all(1)
    rng=np.random.default_rng(2026102609)
    vi=np.sort(rng.choice(np.flatnonzero(full&~data['train']),100,replace=False))
    available=full&data['train']&(ids<3100000000)
    ti=[]
    for v in vi:
        candidates=np.flatnonzero(available&(data['cohort']==data['cohort'][v]))
        t=candidates[np.argmin(np.abs(rms[candidates]-rms[v]))]
        ti.append(t);available[t]=False
    ti=np.asarray(ti);rows=np.r_[ti,vi]
    run.exp.atomic(HERE/'selection.json',dict(selection_seed=2026102609,
        validation_selection='100random original validation cases with all7 bodies active',
        training_selection='100unique original training cases, same source cohort, nearest whole-pose RMS per validation case',
        train_ids=ids[ti].tolist(),validation_ids=ids[vi].tolist(),
        maximum_paired_rms_difference=float(np.abs(rms[ti]-rms[vi]).max()),
        training=describe(rms[ti]),validation=describe(rms[vi]),
        selected_before_prediction=True,added4000_training_cases_in_this_probe=False))
    saved=torch.load(run.HERE/'last.pt',map_location='cpu',weights_only=False)
    stats=dict(mean=saved['model']['extra_mean'].numpy(),std=saved['model']['extra_std'].numpy())
    model=run.make_model(system,stats);model.load_state_dict(saved['model'],strict=True);model.eval()
    center=system.legacy.NikonTensorOptics(nodes=system.legacy.nikon35.AUDIT_NODES,fields=(0.,),device='cuda')
    outer=system.legacy.NikonTensorOptics(nodes=system.legacy.nikon35.NODES,fields=run.EXTRA_FIELDS,device='cuda')
    with np.load(run.HERE/'evaluation6000/final_validation6000.npz') as z:
        lookup={int(identity):i for i,identity in enumerate(z['original_ids'])}
        oldrows=np.asarray([lookup[int(identity)] for identity in ids[vi]])
        val_command=z['command'][oldrows].copy()
        val_fields=z['fields'][oldrows].copy();val_valid=z['valid'][oldrows].copy()
    def extra_wave(physical):
        c=center_wave(center,physical)
        t=outer.trace(physical.double()+outer.bias)
        d=(t['opl'][:,:,1:]-outer.nominal_opl)@outer.projector.T/outer.wave
        return torch.cat((c,d.flatten(1)),1)
    def fit(physical,raw,extra):
        native=model.difference(physical)*7.
        added=extra_wave(physical)
        r=(native-raw.flatten(1).double()).reshape(-1,3,49)
        e=added-extra.double()
        return dict(native=r.square().mean((1,2)).sqrt(),
            center_abs=e[:,:197].square().mean(1).sqrt(),
            outer_diff=e[:,197:].reshape(-1,4,49).square().mean((1,2)).sqrt())
    run.exp.atomic(HERE/'state.json',dict(status='diagnosing',cases=200,optimizer_updates=0))
    with torch.no_grad():
        image=torch.as_tensor(data['image'][rows],device='cuda')
        raw=torch.as_tensor(data['raw'][rows],device='cuda')
        extra=torch.as_tensor(np.array(data['extra'][rows],copy=True),device='cuda')
        target=torch.as_tensor(q[rows],dtype=torch.float64,device='cuda')
        neural=run.exp.LargerNetwork.forward(model,image,raw)
        correction=model.extra_branch((extra-model.extra_mean)/model.extra_std)
        prior=((neural+correction)*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5).double()
        model.trajectory_trace=[]
        estimate=run.four_steps(prior[:100],raw[:100].flatten(1).double()/7.,model)
        # Match the deployed normalized-FP32 output decoding once.
        normalized=((estimate-model.base.target_mean)/model.base.target_std).float()
        train_command=-(normalized*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5)
        trace=model.trajectory_trace;model.trajectory_trace=None
        command=torch.cat((train_command,torch.as_tensor(val_command,device='cuda')),0)
        train_fields,train_valid,_=system.metrics(target[:100]+train_command.double(),target[:100])
        final_fields=np.r_[train_fields.cpu().numpy(),val_fields]
        final_valid=np.r_[train_valid.cpu().numpy(),val_valid]
        prior_fields,prior_valid,_=system.metrics(target-prior,target)
        prior_fields=prior_fields.cpu().numpy();prior_valid=prior_valid.cpu().numpy()
        fitting={name:{key:value.cpu().numpy() for key,value in fit(physical,raw,extra).items()}
                 for name,physical in [('truth_diagnostic_only',target),('neural_prior',prior),('after4',-command.double())]}
        pose_prior=(prior-target).square().mean(1).sqrt().cpu().numpy()
        pose_after=(-command.double()-target).square().mean(1).sqrt().cpu().numpy()
        correction_rms=(correction*model.base.target_std).square().mean(1).sqrt().cpu().numpy()
    report=dict(scope=dict(training_cases=100,validation_cases=100,all_bodies_active=7,
        recovered_network_updates=0,train_final_predictions_computed=100,
        validation_final_predictions_reused=100,neural_prior_predictions=200,
        measurement_acquisitions=1,model_only_steps=4,independent_review=False,
        hidden_pose_used_only_for_diagnostic_scoring_and_noise_floor=True),
        matching=json.loads((HERE/'selection.json').read_text()),groups={})
    for name,section in [('training',slice(0,100)),('validation',slice(100,200))]:
        passed=final_valid[section]&(final_fields[section].max(1)<.07)
        pp=prior_valid[section]&(prior_fields[section].max(1)<.07)
        noise=fitting['truth_diagnostic_only']['native'][section]
        fit4=fitting['after4']['native'][section]
        report['groups'][name]=dict(count=100,passes=int(passed.sum()),prior_passes=int(pp.sum()),
            center_passes=int((final_valid[section]&(final_fields[section,0]<.07)).sum()),
            prior_fields_mean=prior_fields[section].mean(0).tolist(),
            final_fields_mean=final_fields[section].mean(0).tolist(),
            pose_rms_prior=describe(pose_prior[section]),pose_rms_after4=describe(pose_after[section]),
            additional_branch_correction_rms=describe(correction_rms[section]),
            sensor_fit={stage:{channel:describe(a[section]) for channel,a in channels.items()}
                        for stage,channels in fitting.items()},
            native_fit_within_2x_noise_cases=int((fit4<=2*noise).sum()),
            native_fit_within_2x_noise_but_failed=int(((fit4<=2*noise)&~passed).sum()),
            native_fit_median_noise_ratio=float(np.median(fit4/noise)))
    report['training_four_step_trace']=[dict(step=i+1,
        accepted=int((t['gain']>0).sum()),cap_active=int(t['cap_active'].sum()),
        gains={str(float(g)):int((t['gain']==g).sum()) for g in np.unique(t['gain'])}) for i,t in enumerate(trace)]
    report['active_seconds']=run.exp.active_time()-tick
    report['updated']=run.exp.stamp()
    run.exp.atomic(HERE/'result.json',report)
    values=dict(ids=ids[rows],is_train=np.r_[np.ones(100,bool),np.zeros(100,bool)],
        original_q=q[rows],command=command.cpu().numpy(),prior=prior.cpu().numpy(),
        fields=final_fields,valid=final_valid,prior_fields=prior_fields,
        pose_rms_prior=pose_prior,pose_rms_after4=pose_after)
    for stage,channels in fitting.items():
        for channel,a in channels.items():values[f'{stage}_{channel}_fit']=a
    np.savez(HERE/'values.npz',**values)
    run.exp.atomic(HERE/'state.json',dict(status='completed',training_updates=0,
        train_passes=report['groups']['training']['passes'],
        validation_passes=report['groups']['validation']['passes'],active_seconds=report['active_seconds']))
    print(json.dumps({k:v for k,v in report.items() if k!='matching'},indent=2))


if __name__=='__main__':
    main()
