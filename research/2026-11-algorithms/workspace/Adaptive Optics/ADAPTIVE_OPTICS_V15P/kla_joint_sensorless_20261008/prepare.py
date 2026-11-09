"""One-time prescription registration. Never imports old training jobs."""
import os,sys,json,hashlib
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from kla_catadioptric_engine import KLACatadioptricLens
e=KLACatadioptricLens();surfaces=[];z=0.;direction=1
for i,s in enumerate(e.surfaces):
 surfaces.append(dict(s,idx=i,z=z,zdir_in=direction,aperture_mm=1000.))
 z+=s['t']
 if s['mat']=='mirror':direction=-direction
spec=dict(source='https://patents.google.com/patent/US6842298B1/en#description Table 1',surfaces=surfaces,groups=[[i,i+1] for i in range(15,31,2)],n_silica=e.n_silica,pupil_radius_mm=6.7,entrance_obscuration_radius_fraction=.05,image_z=z,field_image_y_mm=np.linspace(-.132,.132,9).tolist(),angles_rad=[0.]*9,reference_points_mm=[[0,0,z]]*9,measured_na=.72)
(HERE/'prescription.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
import common,optics
def chief(a):
 p,d,_,_=optics.numpy_trace(np.zeros((1,2)),a,apertures=False)
 return p[0]+(z-p[0,2])/d[0,2]*d[0]
angles=[brentq(lambda a:chief(a)[1]-h,-.05,.05,xtol=1e-14) for h in spec['field_image_y_mm']]
refs=[chief(a).tolist() for a in angles];spec['angles_rad']=angles;spec['reference_points_mm']=refs
theta=np.arange(720)*2*np.pi/720;xy=np.concatenate([np.c_[np.cos(theta),np.sin(theta)]*r for r in (0.,.05,.25,.5,.75,.9,1.)])
peaks=np.zeros(31)
for a in angles:
 p,d,_,v,hits=optics.numpy_trace(xy,a,apertures=False,history=True)
 assert v.all()
 peaks[1:]=np.maximum(peaks[1:],[np.linalg.norm(h[:,:2],axis=1).max() for h in hits])
for i,s in enumerate(surfaces):
 s['nominal_max_ray_height_mm']=float(peaks[i]);s['aperture_mm']=float(np.ceil((peaks[i]+.30)*100)/100) if i else 6.7
 if s['r']:assert s['aperture_mm']<abs(s['r'])
spec['aperture_definition']='Fixed engineering envelope: nominal 9-field footprints on 720 azimuths x 7 radii plus 0.30mm, rounded upward to 0.01mm. Not measured patent lens diameters. No per-state resize. 5% entrance obscuration models fixed Mangin coating obscuration; coating-hole dimensions unavailable.'
spec['registered_scope']='Existing 6.7mm pupil and +/-0.132mm meridional 9 fields retained as benchmark settings, not asserted Table1 NA0.8 or its patented field specification. Monochromatic 193.3nm scalar pupil model, no vector polarization/coating efficiency.'
p,d,_,_=optics.numpy_trace(np.c_[np.cos(theta),np.sin(theta)],0,apertures=False)
spec['measured_na']=float(np.linalg.norm(d[:,:2],axis=1).max())
spec['chief_plane_span_mm']=refs[-1][1]-refs[0][1]
(HERE/'prescription.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
print(json.dumps(dict(measured_na=spec['measured_na'],angles=angles,span_mm=spec['chief_plane_span_mm'],apertures=[s['aperture_mm'] for s in surfaces],image_z=z)),flush=True)
