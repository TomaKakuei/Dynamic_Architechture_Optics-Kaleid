"""Bounded Nikon mechanism diagnostics, frozen model and saved validation.
No optimizer updates, no independent replay, no full6000 prediction.
"""
from pathlib import Path
import sys,json,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'nikon_sevenbody_diagnosis_20261025'))
import sensor_control as sc
run,np,torch=sc.run,sc.np,sc.torch
from trajectory_backward import TrajectoryStep
LATEST=ROOT/'nikon_prior_supervision_4ep_20261025'


def stats(a):
    a=np.asarray(a)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))


def jac(fun,x):
    with torch.inference_mode(False),torch.no_grad():
        return torch.func.vmap(torch.func.jacfwd(lambda q:fun(q[None])[0]))(x.clone())


def spectrum(j):
    s=torch.linalg.svdvals(j).cpu().numpy()
    return dict(singular_values=s.tolist(),largest=stats(s[:,0]),smallest=stats(s[:,-1]),
                condition=stats(s[:,0]/np.maximum(s[:,-1],1e-30)),
                modes_below_ridge_sqrt=stats((s<.01).sum(1)))


def state(stage,**kw):run.exp.atomic(HERE/'state.json',dict(status=stage,**kw))


def main():
    if (HERE/'result.json').exists():print('Already completed');return
    tick=run.exp.active_time();state('loading')
    system=run.old.make_system('nikon');system.__class__=run.AdditiveSystem
    data=run.old.load_data(system);extra=np.load(run.HERE/'data/extra.npy',mmap_mode='r')
    with np.load(LATEST/'evaluation6000/final_validation6000.npz') as z:saved={k:z[k].copy() for k in z.files}
    active=(np.abs(saved['initial'].reshape(-1,7,5)).max(2)>1e-10).sum(1)
    rng=np.random.default_rng(2026102604)
    slots=np.concatenate([rng.choice(np.flatnonzero((active==7)&(saved['cohort']==c)),16,replace=False) for c in range(3)])
    ids=saved['original_ids'][slots]
    lookup={int(v):i for i,v in enumerate(data['ids'])};rows=np.asarray([lookup[int(v)] for v in ids])
    run.exp.atomic(HERE/'selection.json',dict(seed=2026102604,case_ids=ids.tolist(),
        rule='16random original validation seven-body cases per cohort; before new diagnostics',count=48))
    payload=torch.load(LATEST/'last.pt',map_location='cpu',weights_only=False)['model']
    normal=dict(mean=payload['extra_mean'].numpy(),std=payload['extra_std'].numpy())
    model=run.make_model(system,normal);model.load_state_dict(payload,strict=True);model.eval()
    allsensor=sc.AllSensors(system);native=sc.NativeSensors(system)
    with torch.no_grad():
        q=torch.as_tensor(saved['initial'][slots],device='cuda').double()
        raw=torch.as_tensor(data['raw'][rows],device='cuda').double().flatten(1)/7.
        ex=torch.as_tensor(np.array(extra[rows],copy=True),device='cuda').double()
        measured=torch.cat((raw,ex[:,:197]/np.sqrt(197),ex[:,197:]/7.),1)
        prediction=run.exp.LargerNetwork.forward(model,torch.as_tensor(data['image'][rows],device='cuda'),
                                               torch.as_tensor(data['raw'][rows],device='cuda'))
        correction=model.extra_branch((ex.float()-model.extra_mean)/model.extra_std)
        prior=((prediction+correction)*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5).double()
        estimate=-torch.as_tensor(saved['command'][slots],device='cuda').double()
        zero=torch.zeros_like(q[:1])
        def scoring(estimate):
            f,v,_=system.metrics(q-estimate,q)
            good=v&(f.max(1).values<.07)
            return dict(passes=int(good.sum()),count=len(q),invalid=int((~v).sum()),
                        field_mean=f.mean(0).cpu().tolist(),worst_p90=float(torch.quantile(f.max(1).values,.9)),
                        pose_rms=stats((q-estimate).square().mean(1).sqrt().cpu().numpy())),f.cpu().numpy()
        baseline_fields=saved['fields'][slots];baseline_valid=saved['valid'][slots]
        report=dict(scope=dict(frozen_model=str(LATEST/'last.pt'),training_updates=0,diagnostic_cases=48,
            original6000_not_rerun=True,criterion='CenterABS +/-0.1mm DIFF on197nodes; all3<0.07waves',
            extended_steps='Diagnostic only, not the registered4step result',
            noise_free='Synthetic counterfactual, not current recorded observations'),
            baseline=dict(passes=int((baseline_valid&(baseline_fields.max(1)<.07)).sum()),count=48,
                          field_mean=baseline_fields.mean(0).tolist()),routes={})
        arrays=dict(ids=ids,q=q.cpu().numpy(),prior=prior.cpu().numpy(),estimate4=estimate.cpu().numpy(),fields4=baseline_fields)
        # Continue the saved final estimate, never redo the original4steps.
        clean=native.difference(q)
        for label,opt,obs,start,first in [('native_noisy',native,raw,estimate,5),
                ('all540_noisy',allsensor,measured,estimate,5),('native_clean',native,clean,prior,1)]:
            state('solver_diagnosis',route=label)
            x=start.clone();opt.trajectory_trace=[]
            for step in range(first,17):
                x=TrajectoryStep.apply(x,prior,obs,opt)
                if step in (4,8,16):
                    key=f'{label}_{step}'
                    row,f=scoring(x)
                    row['native_fit_rms_waves']=stats(((native.difference(x)-raw)*7).square().mean(1).sqrt().cpu().numpy())
                    report['routes'][key]=row;arrays[key+'_estimate']=x.cpu().numpy();arrays[key+'_fields']=f
                    run.exp.atomic(HERE/'solver_result.json',report)
            report['routes'][label+'_trace']=[dict(accepted=int((t['gain']>0).sum()),
                capped=int(t['cap_active'].sum())) for t in opt.trajectory_trace]
            opt.trajectory_trace=None
        state('geometry')
        # Sixteen evenly spread examples among the three cohorts.
        small=q[::3];error=(estimate-q)[::3]
        j0=native.response(zero)[0]
        jq=native.response(small)
        ja=allsensor.response(small)
        report['geometry']=dict(native_at_zero=spectrum(j0[None]),native_at_true_pose=spectrum(jq),
            all540_at_true_pose=spectrum(ja))
        u,s,vh=torch.linalg.svd(jq,full_matrices=False)
        coords=(vh@error[:,:,None]).squeeze(-1);weak=s<.01
        weak_error=(vh.transpose(1,2)@(coords*weak)[:,:,None]).squeeze(-1)
        geom=report['geometry']
        geom['fraction_pose_error_in_ridge_damped_modes']=stats((weak_error.square().sum(1)/error.square().sum(1)).cpu().numpy())
        at0=(j0[None]@weak_error[:,:,None]).squeeze(-1).norm(dim=1)
        atq=(jq@weak_error[:,:,None]).squeeze(-1).norm(dim=1)
        geom['same_weak_error_response_amplification_near_zero']=stats((at0/atq.clamp_min(1e-20)).cpu().numpy())
        geom['jacobian_relative_change_from_zero']=stats(((jq-j0).flatten(1).norm(dim=1)/j0.norm()).cpu().numpy())
        def unprojected(physical):
            o=native.native;t=o.trace(physical+o.bias)
            d=(t['opl'][:,:,1:]-o.nominal_opl)/o.wave
            return ((d-d.mean(-1,keepdim=True))/7.).flatten(1)
        jp=jac(unprojected,small)
        geom['retain_tilt_and_defocus_same_fields']=spectrum(jp)
        # Retaining low-order phase is a hypothetical information diagnostic.
        # No such values are fed into the current model or deployment.
        def native_abs(physical):
            m=native.native.metrics(physical)
            return (m['absolute']/7.).flatten(1)
        geom['absolute_reference_sphere_same_fields']=spectrum(jac(native_abs,small))
        # Two additional orthogonal fields, with correctly recomputed nominal OPL.
        cross=run.make_typed_optics(system.legacy.NikonTensorOptics(nodes=system.legacy.nikon35.NODES,
            fields=(0.,-.05,.05,-.05,.05),device='cuda'),torch.float64)
        cross.launch_points[3:,:,1]=cross.launch_points[3:,:,0]
        cross.launch_points[3:,:,0]=0
        cross.nominal_opl.copy_(cross.trace(cross.bias[None])['opl'][0,:,1:])
        def cross_packet(physical):
            t=cross.trace(physical+cross.bias)
            return (((t['opl'][:,:,1:]-cross.nominal_opl)@cross.projector.T/cross.wave)/7.).flatten(1)
        geom['extra_orthogonal_fields']=spectrum(jac(cross_packet,small))
        geom['nonlinearity_by_amplitude']={}
        reference=native.difference(zero)
        for scale in (.25,.5,1.):
            exact=native.difference(small*scale)-reference
            linear=(j0[None]@(small*scale)[:,:,None]).squeeze(-1)
            geom['nonlinearity_by_amplitude'][str(scale)]=stats(((exact-linear).norm(dim=1)/exact.norm(dim=1)).cpu().numpy())
        # Show the same prediction error at the initial pose versus after correction.
        initial_gap=native.difference(estimate)-native.difference(q)
        corrected=native.difference(q-estimate)-reference
        geom['initial_observation_vs_corrected_same_fields']=dict(
            initial_fit_to_noise_free_rms_waves=stats((initial_gap*7).square().mean(1).sqrt().cpu().numpy()),
            correction_residual_rms_waves=stats((corrected*7).square().mean(1).sqrt().cpu().numpy()),
            ratio=stats((corrected.norm(dim=1)/initial_gap.norm(dim=1)).cpu().numpy()))
        arrays.update(native_j0=j0.cpu().numpy(),native_jq=jq.cpu().numpy(),full_jq=ja.cpu().numpy(),
                      loworder_jq=jp.cpu().numpy(),geometry_q=small.cpu().numpy(),geometry_error=error.cpu().numpy())
        report['scales_per_body']=native.native.scales.cpu().tolist()
        report['active_seconds']=run.exp.active_time()-tick
        report['updated']=run.exp.stamp()
        np.savez(HERE/'values.npz',**arrays)
        run.exp.atomic(HERE/'result.json',report);state('completed',active_seconds=report['active_seconds'])
        print(json.dumps(dict(baseline=report['baseline'],routes=report['routes'],seconds=report['active_seconds']),indent=2))


if __name__=='__main__':
    try:main()
    except BaseException:state('failed',error=traceback.format_exc());raise
