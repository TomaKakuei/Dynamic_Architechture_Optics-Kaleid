"""Fit a local-Jacobian surrogate from the original training pairs only.

The exact optical forward model remains available for refinement residuals;
this polynomial is intended to avoid using the nominal Jacobian far from bias.
CPU-only preparation while the separate closed-loop network trial runs.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
start=time.perf_counter()
manifest=json.loads((HERE/'bounded_control/data_manifest.json').read_text())
poses=[];observations=[];seeds=[]
for name in manifest['files']:
    with np.load(name) as z:
        poses.append(z['targets']);observations.append(z['raw']);seeds.append(z['seeds'])
q=np.concatenate(poses).astype(np.float64);raw=np.concatenate(observations).reshape(-1,147).astype(np.float64)
ids=np.concatenate(seeds)
with np.load(HERE/'bounded_control/baseline_validation.npz') as z:
    validation=z['indices'];assert np.array_equal(ids[validation],z['original_ids'])
training=np.ones(len(q),dtype=bool);training[validation]=False
assert training.sum()==24000 and len(validation)==6000
i,j=np.triu_indices(35)
x=np.c_[np.ones(len(q)),q,q[:,i]*q[:,j]]
coefficient,residuals,rank,singular=np.linalg.lstsq(x[training],raw[training],rcond=1e-10)
prediction=x@coefficient
hessian=np.zeros((147,35,35))
for k,(a,b) in enumerate(zip(i,j)):
    if a==b:hessian[:,a,b]=coefficient[36+k]
    else:hessian[:,a,b]=hessian[:,b,a]=coefficient[36+k]/2
np.savez_compressed(HERE/'native_polynomial.npz',intercept=coefficient[0],linear=coefficient[1:36].T,
                    quadratic=hessian,training_indices=np.flatnonzero(training),validation_indices=validation)
def metrics(mask):
    error=(prediction[mask]-raw[mask]).reshape(-1,3,49)
    rms=np.sqrt(np.mean(error**2,axis=(1,2)))
    return dict(count=len(rms),pointwise_rms_waves=float(np.sqrt(np.mean(error**2))),
        sample_rms_median=float(np.median(rms)),sample_rms_p90=float(np.quantile(rms,.9)),
        per_field_rms=np.sqrt(np.mean(error**2,axis=(0,2))).tolist())
report=dict(training=metrics(training),validation=metrics(~training),rank=int(rank),feature_count=x.shape[1],
    seconds=time.perf_counter()-start,scope='Training-only fit of native noisy DIFF measurement versus35pose coordinates; selection-set residual diagnosis',
    planned_use='State-dependent approximate Jacobian; exact nonlinear optics retained for innovation',
    neural_training_updates=0,validation_used_for_fit=False)
(HERE/'native_polynomial_fit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
