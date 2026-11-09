"""Nine actual angular fields; fixed-exposure noisy intensity only.

An explicitly modelled pupil relay maps entrance coordinates onto a 2mm
camera pupil; a fixed diffuser and free propagation produce speckle images.
This is a simulated calibration sensor, not claimed patent hardware.
"""
from optics import *
from camera_integrator import Integrator
from types import SimpleNamespace
class WaveFit:
 def __init__(self,degree=24):
  self.degree=degree;self.terms=[(i,j) for i in range(degree+1) for j in range(degree+1-i)]
  theta=np.arange(128)*2*np.pi/128
  self.xy=np.vstack([disk(41),*[np.c_[np.cos(theta),np.sin(theta)]*r for r in (.05,.94,.98,1.)]])
  self.design=self.basis(self.xy);self.pinv=np.linalg.pinv(self.design,rcond=1e-12);self.optics=Optics(xy=self.xy)
  rng=np.random.default_rng(19336819);t=rng.uniform(0,2*np.pi,256);r=np.sqrt(rng.uniform(.05**2,1,256));self.probe=np.c_[r*np.cos(t),r*np.sin(t)]
  self.check=Optics(xy=self.probe);self.check_design=self.basis(self.probe)
  theta=np.arange(256)*2*np.pi/256;self.unit=np.c_[np.cos(theta),np.sin(theta)];self.radial=np.array([.05,.1,.25,.5,.75,.9,1.]);self.bound=Optics(xy=np.concatenate([self.unit*r for r in self.radial]))
 def basis(self,xy):
  vx=np.polynomial.chebyshev.chebvander(xy[:,0],self.degree);vy=np.polynomial.chebyshev.chebvander(xy[:,1],self.degree)
  return np.column_stack([vx[:,i]*vy[:,j] for i,j in self.terms])
 @torch.no_grad()
 def fit(self,q):
  t=torch.tensor(np.asarray(q)[None],dtype=torch.float64);phase,valid=self.optics.phase(t);truth,ok=self.check.phase(t);_,bv=self.bound.phase(t)
  phase=phase[0].numpy();valid=valid[0].numpy();truth=truth[0].numpy();ok=ok[0].numpy();bv=bv[0].numpy().reshape(9,len(self.radial),256);reps=[]
  for fi in range(9):
   if not bv[fi,0].all() or np.any(np.diff(bv[fi].astype(int),axis=0)>0):raise ValueError('Unsupported pupil topology; halt, do not merit-resample')
   boundary=np.ones(256)
   if not bv[fi,-1].all():
    lo=np.max(np.where(bv[fi],self.radial[:,None],0),axis=0);hi=np.min(np.where(~bv[fi],self.radial[:,None],1),axis=0)
    for _ in range(18):
     mid=(lo+hi)/2;tr=numpy_trace(self.unit*mid[:,None],SPEC['angles_rad'][fi],q);lo=np.where(tr[3],mid,lo);hi=np.where(tr[3],hi,mid)
    boundary=lo
   v=valid[fi]
   # The fit nodes intentionally oversample four pupil rings. Their count is
   # not an area measure. Integrate the already traced radial boundary of
   # this star-shaped annulus instead; keep the physical98% requirement.
   pupil_area_fraction=float(np.mean(boundary**2-.05**2)/(1-.05**2))
   if pupil_area_fraction<.98:raise ValueError(f'Less than98% pupil AREA: {pupil_area_fraction:.9f}; fit-node fraction={v.mean():.9f}')
   coef=self.pinv@phase[fi] if v.all() else np.linalg.lstsq(self.design[v],phase[fi,v],rcond=1e-12)[0]
   residual=self.design[v]@coef-phase[fi,v];error=self.check_design[ok[fi]]@coef-truth[fi,ok[fi]];error-=error.mean()
   er=float(np.sqrt(np.mean(error**2)));fr=float(np.sqrt(np.mean(residual**2)))
   if max(er,fr)>.0003:raise RuntimeError(f'Continuous phase fit failed: {fr},{er}')
   matrix=np.zeros((self.degree+1,self.degree+1))
   for k,(i,j) in enumerate(self.terms):matrix[i,j]=coef[k]
   linear=np.linalg.lstsq(np.c_[np.ones(v.sum()),self.xy[v]],phase[fi,v],rcond=None)[0]
   reps.append(dict(coefficients=matrix,boundary=boundary,carrier=linear[1:],fit_rms_waves=fr,independent_fit_rms_waves=er,pupil_survival_fraction=pupil_area_fraction,fit_node_survival_fraction=float(v.mean())))
  return reps
class Camera(Integrator):
 camera_diameter_mm=2.;propagation_mm=2.;sensor_size=128
 def __init__(self,device='cuda',degree=24,calibration=None):
  self.degree=degree;self.device=torch.device(device);self.cache={};self.aperture_cache={}
  self.engine=SimpleNamespace(diffuser=np.load(HERE.parent/'artifacts/diffuser_mask.npy'),wave=193.3)
  rng=np.random.default_rng(193320261008);self.prnu=rng.normal(1,.005,(128,128));self.dsnu=rng.normal(2,.5,(128,128))
  self.gain_e=None if calibration is None else calibration['gain_e_per_intensity']
 def phase_slope(self,rep):
  # Check the carrier-removed phase bandwidth before rasterizing a capture.
  # A dense polynomial derivative grid includes the illuminated rim. The
  # thresholds leave a margin below Nyquist; hard cases are retained.
  coord=np.linspace(-1,1,65);v=np.polynomial.chebyshev.chebvander(coord,self.degree);x,y=np.meshgrid(coord,coord);mask=x*x+y*y<=1
  coef=rep['coefficients'].copy();coef[1,0]-=rep['carrier'][0];coef[0,1]-=rep['carrier'][1];peak=0.
  for axis in (0,1):
   der=np.polynomial.chebyshev.chebder(coef,axis=axis);values=v[:,:der.shape[1]]@der.T@v[:,:der.shape[0]].T;peak=max(peak,float(abs(values[mask]).max()))
  return peak
 def calibrate(self,reps,n):
  intensity=np.array([self.integrated_irradiance(r,n) for r in reps]);self.gain_e=4000/float(intensity.max())
  return dict(gain_e_per_intensity=self.gain_e,nominal_peak_e=4000.,fixed_exposure=True,prnu_sigma=.005,dsnu_mean_e=2.,dsnu_sigma_e=.5,read_sigma_e=1.2,full_well_e=10000.,adu_per_e=4.,bias_adu=100.,sensor_seed=193320261008,grid=n,degree=self.degree,diffuser_sha256=hashlib.sha256((HERE.parent/'artifacts/diffuser_mask.npy').read_bytes()).hexdigest())
 def noisy(self,intensity,seed):
  if self.gain_e is None:raise RuntimeError('Missing fixed exposure calibration')
  rng=np.random.default_rng(seed);expect=np.maximum(intensity*self.gain_e*self.prnu,0)
  sensed=np.clip(rng.poisson(expect)+rng.normal(0,1.2,expect.shape)+self.dsnu,0,10000)
  return np.rint(sensed*4+100).clip(0,65535).astype(np.uint16)
 def capture(self,reps,seed,n):
  values=[];grids=[]
  for r in reps:
   slope=self.phase_slope(r);needed=2048 if slope<=450 else (4096 if slope<=900 else 8192)
   if slope>1800:raise RuntimeError('Camera bandwidth outside verified raster support; halt rather than alias or resample state')
   grids.append(needed)
   values.append(self.integrated_irradiance(r,n) if needed<=n else self.reference_irradiance(r,needed,carrier_shift=True))
  if max(grids)>n:print(json.dumps(dict(camera_high_resolution_seed=seed,pupil_grids=grids)),flush=True)
  self.last_grids=grids
  return self.noisy(np.array(values),seed)
