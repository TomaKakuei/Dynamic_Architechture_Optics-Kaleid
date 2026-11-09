"""Sensorless intensity: fixed relay, diffuser, exposure, pixel maps and noise.

An ideal pupil relay maps the 6 mm entrance-coordinate wavefront to a 2 mm
sensor pupil; sensor width4 mm, propagation2 mm,128x128 integrated pixels.
This calibrated relay is a simulation component, not a paper instrument claim.
Piston is irrelevant; physical tilt remains in the carrier-shifted propagator.
"""
from optics import *
from camera_integrator import Integrator
from types import SimpleNamespace
from scipy.signal import resample
class WaveFit:
 def __init__(self,degree=16):
  self.degree=degree;self.terms=[(i,j) for i in range(degree+1) for j in range(degree+1-i)]
  theta=np.arange(128)*2*np.pi/128
  self.xy=np.vstack([disk(33),*[np.c_[np.cos(theta),np.sin(theta)]*r for r in (.94,.98,1.)]])
  self.design=self.basis(self.xy);self.pinv=np.linalg.pinv(self.design,rcond=1e-12)
  self.optics=Optics(xy=self.xy)
  rng=np.random.default_rng(6819);t=rng.uniform(0,2*np.pi,192);r=np.sqrt(rng.uniform(0,1,192));self.probe=np.c_[r*np.cos(t),r*np.sin(t)]
  self.check=Optics(xy=self.probe);self.check_design=self.basis(self.probe)
  angles=np.arange(256)*2*np.pi/256;self.boundxy=np.concatenate([np.c_[np.cos(angles),np.sin(angles)]*r for r in (.25,.5,.75,.9,1.)]);self.bound=Optics(xy=self.boundxy)
 def basis(self,xy):
  vx=np.polynomial.chebyshev.chebvander(xy[:,0],self.degree);vy=np.polynomial.chebyshev.chebvander(xy[:,1],self.degree)
  return np.column_stack([vx[:,i]*vy[:,j] for i,j in self.terms])
 @torch.no_grad()
 def fit(self,q):
  t=torch.tensor(np.asarray(q)[None],dtype=torch.float64);phase,valid=self.optics.phase(t);truth,ok=self.check.phase(t);_,bv=self.bound.phase(t)
  phase=phase[0].numpy();valid=valid[0].numpy();truth=truth[0].numpy();ok=ok[0].numpy();bv=bv[0].numpy().reshape(9,5,256)
  reps=[]
  for fi in range(9):
   if not bv[fi,0].all() or np.any(np.diff(bv[fi].astype(int),axis=0)>0):raise ValueError('unsupported non-star-shaped pupil')
   if not bv[fi,-1].all():
    # Precise radial boundary in fixed source coordinates; do not zero-fill phase.
    theta=np.arange(256)*2*np.pi/256;unit=np.c_[np.cos(theta),np.sin(theta)];lo=np.max(np.where(bv[fi],np.array([.25,.5,.75,.9,1.])[:,None],0),axis=0);hi=np.min(np.where(~bv[fi],np.array([.25,.5,.75,.9,1.])[:,None],1),axis=0)
    for _ in range(18):
     mid=(lo+hi)/2;tr=numpy_trace(unit*mid[:,None],FIELDS[fi],q);lo=np.where(tr[3],mid,lo);hi=np.where(tr[3],hi,mid)
    boundary=lo
   else:boundary=np.ones(256)
   v=valid[fi]
   if v.mean()<.98:raise ValueError('less than98% pupil support')
   coef=self.pinv@phase[fi] if v.all() else np.linalg.lstsq(self.design[v],phase[fi,v],rcond=1e-12)[0]
   fit=self.design[v]@coef-phase[fi,v];error=self.check_design[ok[fi]]@coef-truth[fi,ok[fi]];error-=error.mean()
   er=float(np.sqrt(np.mean(error**2)));fr=float(np.sqrt(np.mean(fit**2)))
   if max(er,fr)>.0003:raise RuntimeError(f'continuous phase fit failed: {fr},{er}')
   matrix=np.zeros((self.degree+1,self.degree+1))
   for k,(i,j) in enumerate(self.terms):matrix[i,j]=coef[k]
   linear=np.linalg.lstsq(np.c_[np.ones(v.sum()),self.xy[v]],phase[fi,v],rcond=None)[0]
   reps.append(dict(coefficients=matrix,boundary=boundary,carrier=linear[1:],fit_rms_waves=fr,independent_fit_rms_waves=er,pupil_survival_fraction=float(v.mean())))
  return reps
class Camera(Integrator):
 camera_diameter_mm=2.;propagation_mm=2.;sensor_size=128
 def __init__(self,device='cuda',degree=16,calibration=None):
  self.degree=degree;self.device=torch.device(device);self.cache={};self.aperture_cache={}
  mask=np.load(HERE.parent/'artifacts/diffuser_mask.npy')
  self.engine=SimpleNamespace(diffuser=mask,wave=910.)
  rng=np.random.default_rng(91020261008);self.prnu=rng.normal(1,.005,(128,128));self.dsnu=rng.normal(2,.5,(128,128))
  self.gain_e=None if calibration is None else calibration['gain_e_per_intensity']
 def calibrate(self,reps,n):
  intensity=np.array([self.integrated_irradiance(r,n) for r in reps]);self.gain_e=4000/float(intensity.max())
  return dict(gain_e_per_intensity=self.gain_e,nominal_peak_e=4000.,fixed_exposure=True,prnu_sigma=.005,dsnu_mean_e=2.,dsnu_sigma_e=.5,read_sigma_e=1.2,full_well_e=10000.,adu_per_e=4.,bias_adu=100.,sensor_seed=91020261008,grid=n,diffuser_sha256=hashlib.sha256((HERE.parent/'artifacts/diffuser_mask.npy').read_bytes()).hexdigest())
 def noisy(self,intensity,seed):
  if self.gain_e is None:raise RuntimeError('Missing fixed nominal exposure calibration')
  rng=np.random.default_rng(seed);expect=np.maximum(intensity*self.gain_e*self.prnu,0)
  sensed=np.clip(rng.poisson(expect)+rng.normal(0,1.2,expect.shape)+self.dsnu,0,10000)
  return np.rint(sensed*4+100).clip(0,65535).astype(np.uint16)
 def capture(self,reps,seed,n):
  intensities=np.array([self.integrated_irradiance(r,n) for r in reps])
  return self.noisy(intensities,seed)
