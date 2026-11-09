"""Isolated KLA angular-field, noisy-camera, joint-controller run."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import json,time,hashlib,ctypes,uuid
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'prescription.json').read_text(encoding='utf-8'))
GROUPS=SPEC['groups'];SCALES=np.array([.08,.08,.08,2.5/60,2.5/60])
FIELDS=np.c_[np.zeros(9),np.linspace(-.132,.132,9)]
WAVE=.0001933
TRAIN=9000;VALIDATION=1000;PER_COHORT=TRAIN+VALIDATION;TOTAL=3*PER_COHORT
def stamp():
 from datetime import datetime,timezone
 return datetime.now(timezone.utc).isoformat()
def atomic(path,obj):
 path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
 tmp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');os.replace(tmp,path)
def clock():
 if os.name=='nt':
  t=ctypes.c_ulonglong();ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(t));return t.value/1e7
 return time.monotonic()
def awake(on=True):
 if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001 if on else 0x80000000)
def hashes():
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir()) if p.suffix=='.py' or p.name in ('prescription.json','protocol.json')}
def disk(grid,inner=.05):
 y,x=np.mgrid[-1:1:complex(grid),-1:1:complex(grid)];m=(x*x+y*y<=1)&(x*x+y*y>=inner*inner)
 return np.c_[x[m],y[m]]
def rotation(q):
 tx,ty=np.deg2rad(q);cx,sx,cy,sy=np.cos(tx),np.sin(tx),np.cos(ty),np.sin(ty)
 return np.array([[cy,sy*sx,sy*cx],[0,cx,-sx],[-sy,cy*sx,cy*cx]])
def sag(r,x):
 x=np.asarray(x);return np.zeros_like(x) if r==0 else x*x/(r+np.copysign(np.sqrt(np.maximum(r*r-x*x,0)),r))
