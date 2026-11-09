"""Versioned Trepan run: no imports or writes to the obsolete run."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[k]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import json,time,hashlib,ctypes,uuid
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
SPEC=json.loads((HERE/'prescription.json').read_text(encoding='utf-8'))
GROUPS=SPEC['groups'];SCALES=np.array([.1,.1,.1,2.5/60,2.5/60])
FIELDS=np.asarray(SPEC['fields'])
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
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir()) if p.suffix in ('.py',) or p.name in ('prescription.json','protocol.json')}
def disk(grid):
 y,x=np.mgrid[-1:1:complex(grid),-1:1:complex(grid)];m=x*x+y*y<=1
 return np.c_[x[m],y[m]]
def rotation(q):
 tx,ty=np.deg2rad(q);cx,sx,cy,sy=np.cos(tx),np.sin(tx),np.cos(ty),np.sin(ty)
 return np.array([[cy,sy*sx,sy*cx],[0,cx,-sx],[-sy,cy*sx,cy*cx]])
