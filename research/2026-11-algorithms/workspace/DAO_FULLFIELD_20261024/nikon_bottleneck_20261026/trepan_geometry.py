"""Read Trepan saved outcomes and compare optical geometry; no Trepan training.
Phase geometry is not a camera Fisher-information or noise comparison.
"""
from pathlib import Path
import os,sys,json,traceback
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
SOURCE=ROOT.parent/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/trepan_restored_joint_20261008'
sys.path.insert(0,str(SOURCE))
from optics import Optics
from common import FIELDS,SCALES,clock
import numpy as np
import torch


def stats(a):
    a=np.asarray(a);return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),max=float(a.max()))


def jac(fun,x):
    with torch.no_grad():return torch.func.vmap(torch.func.jacfwd(lambda q:fun(q[None])[0]))(x.clone())


def spectrum(j):
    s=torch.linalg.svdvals(j).cpu().numpy()
    return dict(singular_values=s.tolist(),largest=stats(s[:,0]),smallest=stats(s[:,-1]),
        modes_below_0p01=stats((s<.01).sum(1)),condition=stats(s[:,0]/np.maximum(s[:,-1],1e-30)))


def main():
    path=HERE/'trepan_geometry.json'
    if path.exists():print('Already completed');return
    tick=clock()
    with np.load(ROOT/'trepan_large/validation_037.npz') as z:d={k:z[k].copy() for k in z.files}
    with np.load(ROOT/'nikon_prior_supervision_4ep_20261025/evaluation6000/final_validation6000.npz') as z:
        nikon_q=z['initial'].copy()
    qfull=d['initial']
    rows=np.random.default_rng(2026102605).choice(len(qfull),16,replace=False)
    q=torch.as_tensor(qfull[rows],device='cuda').double();zero=torch.zeros_like(q[:1])
    optics=Optics(grid=17,device='cuda')
    n=len(optics.xy)
    def sensor(x):
        p,v=optics.phase(x)
        return (p*v/n**.5).flatten(1)
    def pttd(x):
        phase,valid=optics.phase(x);w=valid.to(phase.dtype);a=optics.design
        gram=torch.einsum('nr,bfn,ns->bfrs',a,w,a)
        rhs=torch.einsum('nr,bfn->bfr',a,w*phase)
        coef=torch.linalg.solve(gram+torch.eye(4,device=x.device,dtype=x.dtype)*1e-12,rhs[...,None])[...,0]
        return ((phase-torch.einsum('nr,bfr->bfn',a,coef))*w/n**.5).flatten(1)
    with torch.no_grad():
        j0=jac(sensor,zero)[0];jq=jac(sensor,q);jp=jac(pttd,q)
        baseline=sensor(zero);actual=sensor(q)-baseline;linear=(j0[None]@q[:,:,None]).squeeze(-1)
        support=optics.phase(q)[1].double().mean(-1)
        endpoint=torch.as_tensor((qfull+d['command'])[rows],device='cuda').double()
        corrected_jac=jac(sensor,endpoint)
    activity=(np.abs(qfull.reshape(-1,8,5)).max(2)>1e-10).sum(1)
    mask=(np.abs(nikon_q.reshape(-1,7,5)).max(2)>1e-10).sum(1)==7
    report=dict(scope='16 saved Trepan validation states;197-node ideal optical phase geometry, not actual camera observability/noise',
        seed=2026102605,rows=rows.tolist(),fields_degrees=FIELDS.tolist(),scales=SCALES.tolist(),
        phase_information='Camera retains tilt and defocus; Nikon stored DIFF maps remove piston/tilt/defocus',
        phase_sensor=spectrum(jq),phase_at_zero=spectrum(j0[None]),pttd_only=spectrum(jp),
        relative_jacobian_change_from_zero=stats(((jq-j0).flatten(1).norm(dim=1)/j0.norm()).cpu().numpy()),
        linearization_error=stats(((actual-linear).norm(dim=1)/actual.norm(dim=1)).cpu().numpy()),
        relative_jacobian_change_to_corrected_endpoint=stats(((jq-corrected_jac).flatten(1).norm(dim=1)/corrected_jac.flatten(1).norm(dim=1)).cpu().numpy()),
        minimum_pupil_support=float(support.min()),
        saved_validation=dict(count=len(qfull),all9_below08=int((d['valid']&(d['fields'].max(1)<.08)).sum()),
                              all9_below07=int((d['valid']&(d['fields'].max(1)<.07)).sum())),
        sampling=dict(trepan_all8_count=int((activity==8).sum()),trepan_normalized_pose_rms=stats(np.sqrt((qfull*qfull).mean(1))),
            nikon_full7_normalized_pose_rms=stats(np.sqrt((nikon_q[mask]**2).mean(1))),
            different_physical_translation_scales=True),
        active_seconds=clock()-tick,training_updates=0,new_network_predictions=0)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez(HERE/'trepan_geometry_arrays.npz',q=q.cpu().numpy(),j0=j0.cpu().numpy(),jq=jq.cpu().numpy(),jpttd=jp.cpu().numpy())
    print(json.dumps({k:v for k,v in report.items() if k not in ('phase_sensor','phase_at_zero','pttd_only')},indent=2))


if __name__=='__main__':main()
