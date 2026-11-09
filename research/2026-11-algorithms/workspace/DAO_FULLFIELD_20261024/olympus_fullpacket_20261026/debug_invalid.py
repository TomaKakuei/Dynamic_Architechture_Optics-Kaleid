"""Bounded failure diagnosis, not an evaluation replay or a new full run."""
from support import *
from camera_inverse import CameraModel
import json,os,traceback

OUT=HERE/'debug_invalid';OUT.mkdir(exist_ok=True)


def log(stage,**kw):
    common.atomic(OUT/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))
    print(json.dumps(dict(stage=stage,**kw)),flush=True)


@torch.no_grad()
def main():
    with np.load(SOURCE/'training/selected_validation.npz') as z:old={k:z[k].copy() for k in z.files}
    summary=json.loads((HERE/'stopped_summary.json').read_text())
    # Fixed failure/control examples, chosen from already saved outcomes.
    cases=[7007,7111,7049,7363]
    common.atomic(OUT/'protocol.json',dict(cases=cases,scope='Failure-conditioned local mechanism diagnosis; not unbiased evaluation',
        controls='Saved prior and final command; local image derivative; true pose forward rendering used only to diagnose model mismatch',
        acquisitions=0,training_updates=0,full_evaluation_resume=False))
    log('initializing',cases=cases)
    camera=CameraModel();blocks=Blocks();results=[]
    for case in cases:
        dest=OUT/f'case_{case}.json'
        if dest.exists():results.append(json.loads(dest.read_text()));continue
        slot=int(np.where(old['index']==case)[0][0]);initial=old['initial'][slot]
        saved=json.loads((HERE/'evaluation500'/f'case_{case}.json').read_text())
        prior=-old['command'][slot];final=-np.asarray(saved['command']);drift=final-prior
        endpoint=initial-final
        images,_=blocks.read('joint',case//25);image=torch.as_tensor(images[case%25],device='cuda').float()
        sigma=sigma_from_adu(image);tick=common.clock()
        row=dict(case=case,prior=prior.tolist(),final=final.tolist(),endpoint=endpoint.tolist(),
            exceeded_coordinates=np.where(abs(endpoint)>1)[0].tolist(),
            estimate_from_nominal=camera.camera.engine.geometry.path(final),
            max_drift=float(abs(drift).max()),pose_rms_before=float(np.sqrt(np.mean((initial-prior)**2))),
            pose_rms_after=float(np.sqrt(np.mean(endpoint**2))))
        if abs(endpoint).max()<=1:
            row['actual_correction_path']=camera.camera.engine.geometry.path(endpoint,start=initial)
            row['endpoint_clearance_mm']=camera.camera.engine.geometry.clearance_bounds(endpoint).tolist()
            row['clearance_names']=camera.camera.engine.geometry.describe()['constraint_names']
            _,pupil,chief=camera.optics.phase(torch.as_tensor(endpoint[None],device='cuda').double())
            row['endpoint_chief_valid']=chief[0].tolist();row['endpoint_min_pupil']=float(pupil[0].double().mean(-1).min())
        log('local_response',case=case)
        pred,j=camera.evaluate(prior,True)
        residual=((pred-image)/sigma/128).flatten().double()
        J=(j/sigma[...,None]/128).reshape(-1,35).double()
        gram=J.T@J;rhs=J.T@residual
        ev,V=torch.linalg.eigh(gram);ev=ev.clamp_min(0);s=ev.sqrt()
        delta=-torch.linalg.solve(gram+1e-4*torch.eye(35,device='cuda',dtype=torch.float64),rhs)
        drift_t=torch.as_tensor(drift,device='cuda').double();mode=V.T@drift_t
        weak=(s<1/(128*.1))
        row.update(singular_values_ascending=s.cpu().tolist(),
            ridge=1e-4,regularizer_cost_at_final=1e-4*float(drift@drift),
            standardized_pixel_cost_prior=float(residual.square().sum()),
            raw_first_delta_max=float(delta.abs().max()),
            weak_modes_1sigma_pose_above_point1=int(weak.sum()),
            drift_energy_fraction_in_those_modes=float(mode[weak].square().sum()/mode.square().sum()),
            curvature_condition=float(ev[-1]/ev[0].clamp_min(1e-30)))
        # Directional finite difference of the same model isolates the local
        # derivative approximation (not a re-scoring of full trajectories).
        direction=drift/max(float(abs(drift).max()),1e-12)
        analytic=J@torch.as_tensor(direction,device='cuda').double()
        fdresults=[]
        for h in (1e-3,3e-3):
            if max(abs(prior+h*direction).max(),abs(prior-h*direction).max())>1:
                fdresults.append(dict(h=h,skipped='outside_model_box'));continue
            plus,_=camera.evaluate(prior+h*direction);minus,_=camera.evaluate(prior-h*direction)
            fd=((plus-minus)/(2*h)/sigma/128).flatten().double()
            fdresults.append(dict(h=h,norm_fd=float(fd.norm()),norm_analytic=float(analytic.norm()),
                relative_error=float((fd-analytic).norm()/fd.norm()),
                cosine=float(torch.nn.functional.cosine_similarity(fd,analytic,dim=0))))
        row['directional_image_derivative']=fdresults
        log('model_mismatch',case=case)
        # Known simulator pose is a diagnostic oracle ONLY; never passed into
        # the candidate, used for step selection, or turned into new input data.
        expected_true,_=camera.evaluate(initial)
        residual_true=((expected_true-image)/sigma/128).flatten().double()
        row['standardized_pixel_cost_at_true_pose']=float(residual_true.square().sum())
        row['phase_fit_at_true_pose']=camera.last_diagnostics
        # Gradient components attributable locally to residual at the true pose.
        miss_delta=-torch.linalg.solve(gram+1e-4*torch.eye(35,device='cuda',dtype=torch.float64),J.T@residual_true)
        row['local_step_from_true_pose_image_residual_max']=float(miss_delta.abs().max())
        row['local_step_from_true_pose_image_residual_norm']=float(miss_delta.norm())
        expected_final,_=camera.evaluate(final)
        row['standardized_pixel_cost_final']=float((((expected_final-image)/sigma/128).square().sum()))
        row['active_seconds']=common.clock()-tick
        common.atomic(dest,row);results.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('prior','final','endpoint','phase_fit_at_true_pose','singular_values_ascending','endpoint_clearance_mm','clearance_names')}),flush=True)
    common.atomic(OUT/'summary.json',dict(rows=results,stopped_cases=summary['cases'],no_new_inference_trajectory=True))
    log('completed',cases=len(results))


if __name__=='__main__':
    try:main()
    except BaseException:log('failed',error=traceback.format_exc());raise
