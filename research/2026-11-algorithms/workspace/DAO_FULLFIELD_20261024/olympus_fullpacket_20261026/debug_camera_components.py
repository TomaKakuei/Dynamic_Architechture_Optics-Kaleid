"""Isolate model mismatch and missing aperture response on two saved states."""
from support import *
from camera_inverse import CameraModel
import json
OUT=HERE/'debug_invalid'


@torch.no_grad()
def main():
    out=OUT/'camera_components.json'
    if out.exists():print(out.read_text());return
    with np.load(SOURCE/'training/selected_validation.npz') as z:old={k:z[k].copy() for k in z.files}
    camera=CameraModel();answer={}
    for case in (7007,7049):
        slot=int(np.where(old['index']==case)[0][0]);q=old['initial'][slot]
        with np.load(SOURCE/'data'/f'batch_{case//25:04d}.npz') as z:
            image=torch.as_tensor(z['speckles'][case%25].copy(),device='cuda').float()
            degree=int(z['phase_fit_degree'][case%25]);target=z['targets_normalized'][case%25].copy()
        sigma=sigma_from_adu(image);rows=camera.representations(q,False)
        predictions=torch.stack([camera.field(row,False)[0] for row in rows])
        costs=(((predictions-image)/sigma)**2).mean((1,2))
        fi=int(costs.argmax());r=dict(case=case,stored_phase_degree=degree,
            target_label_gap=float(abs(q-target).max()),costs_by_field=costs.tolist(),field_examined=fi)
        raw,valid=camera.camera.phase_at(q,fi,camera.camera.xy)
        tensor,mask,_=camera.phase(torch.as_tensor(q[None],device='cuda').double())
        tensor=tensor[0,fi].cpu().numpy();mask=mask[0,fi].cpu().numpy()
        common_mask=mask&valid;difference=tensor[common_mask]-raw[common_mask];difference-=difference.mean()
        r.update(raw_phase_difference_rms=float(np.sqrt(np.mean(difference**2))),
                 raw_phase_difference_max=float(abs(difference).max()),mask_disagreement=int((mask!=valid).sum()))
        original_rows=camera.camera.wave_representation(q,verify=False)
        ref=original_rows[fi]
        r['original_degree24_fit_rms']=ref['fit_rms_waves']
        r['fit_matrix_difference_max']=float(abs(ref['coefficients']-rows[fi]['matrix'].cpu().numpy()).max())
        r['carrier_difference']=(np.asarray(ref['carrier'])-rows[fi]['carrier'].cpu().numpy()).tolist()
        for n in (1024,2048):
            inten=camera.camera.integrate_field(ref,n)
            expected=torch.as_tensor(camera.camera.expected_adu(inten),device='cuda')
            r[f'original_phase_render_{n}_cost']=float((((expected-image[fi])/sigma[fi])**2).mean())
            if n==1024:r['original_vs_new_image_rms_adu']=float((expected-predictions[fi]).square().mean().sqrt())
        if degree!=24:
            accurate=RelayCamera(device='cuda',degree=degree)
            rr=accurate.wave_representation(q,verify=False)[fi]
            expected=torch.as_tensor(accurate.expected_adu(accurate.integrate_field(rr,2048)),device='cuda')
            r['recorded_degree_full_resolution_cost']=float((((expected-image[fi])/sigma[fi])**2).mean())
            del accurate
        answer[str(case)]=r;common.atomic(out,answer);print(json.dumps(r),flush=True)
    # Same ray phase, with or without the aperture changing in the FD response.
    case=7111;slot=int(np.where(old['index']==case)[0][0]);q=-old['command'][slot]
    saved=json.loads((HERE/'evaluation500'/f'case_{case}.json').read_text());d=-np.array(saved['command'])-q;d/=abs(d).max()
    with np.load(SOURCE/'data'/f'batch_{case//25:04d}.npz') as z:image=torch.as_tensor(z['speckles'][case%25].copy(),device='cuda').float()
    sigma=sigma_from_adu(image)
    central=camera.representations(q,True)
    jac=torch.stack([camera.field(row,True)[1] for row in central])
    analytic=(jac/sigma[...,None]/128).reshape(-1,35).double()@torch.as_tensor(d,device='cuda').double()
    derivatives=[]
    for h in (.003,.01,.03):
        if max(abs(q+h*d).max(),abs(q-h*d).max())>1:continue
        plus=camera.representations(q+h*d,False);minus=camera.representations(q-h*d,False)
        before=[];after=[];fixedplus=[];fixedminus=[]
        for fi in range(9):
            before.append(camera.field(plus[fi],False)[0]);after.append(camera.field(minus[fi],False)[0])
            plus[fi]['boundary']=central[fi]['boundary'];minus[fi]['boundary']=central[fi]['boundary']
            fixedplus.append(camera.field(plus[fi],False)[0]);fixedminus.append(camera.field(minus[fi],False)[0])
        for kind,p,m in [('full',before,after),('frozen_boundary',fixedplus,fixedminus)]:
            fd=((torch.stack(p)-torch.stack(m))/(2*h)/sigma/128).flatten().double()
            derivatives.append(dict(h=h,kind=kind,fd_norm=float(fd.norm()),analytic_norm=float(analytic.norm()),
                relative_error=float((fd-analytic).norm()/fd.norm()),cosine=float(torch.nn.functional.cosine_similarity(fd,analytic,dim=0))))
    answer['aperture_derivative_7111']=derivatives;common.atomic(out,answer);print(json.dumps(derivatives),flush=True)


if __name__=='__main__':main()
