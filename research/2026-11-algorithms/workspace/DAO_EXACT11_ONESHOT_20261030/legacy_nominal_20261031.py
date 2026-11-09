"""Original historical scorer, nominal state, no simulated measurement noise."""
from pathlib import Path
import sys,os,json
HERE=Path(__file__).resolve().parent
sys.pycache_prefix=str(HERE/'local_pycache')
env=Path(sys.prefix);sys.path.append(str(env/'Lib/site-packages'))
_dll=os.add_dll_directory(str(env/'Library/bin'))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
source=HERE.parent/'Adaptive Optics'
sys.path.insert(0,str(source))
import numpy as np
from kla_full_fov_engine import KLAFullFOVEngine
engine=KLAFullFOVEngine()
fields=[];frequency=[]
for y in engine.field_points_y:
    r=engine.get_field_opd_map(y,{},grid_size=64,add_noise=False)
    assert r is not None
    k=int(np.argmin(abs(r['freqs']-2000.)))
    fields.append(float(r['mtf_tangential'][k]));frequency.append(float(r['freqs'][k]))
a=np.array(fields)
result=dict(scope='Historical original64-grid scorer, zero perturbations, no noise, no corrections',
    source=str(source/'kla_full_fov_engine.py'),grid=64,fields=fields,frequency_lpmm=frequency,
    MTF_mean=float(a.mean()),MTF_std=float(a.std()),MTF_min=float(a.min()),Q2000=float(a.mean()-.3*a.std()))
(HERE/'legacy_nominal_20261031.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
