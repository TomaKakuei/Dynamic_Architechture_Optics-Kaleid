"""Zero perturbation/command, unchanged current KLA optical prescription."""
from pathlib import Path
import sys,os,json,hashlib,time
HERE=Path(__file__).resolve().parent
sys.pycache_prefix=str(HERE/'local_pycache')
env=Path(sys.prefix)
sys.path.append(str(env/'Lib/site-packages'))
_dll=os.add_dll_directory(str(env/'Library/bin'))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
source=HERE.parent/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0,str(source))
from optics import Optics,torch,np
torch.set_num_threads(1)
rows=[]
for grid in (65,129,257):
    start=time.perf_counter()
    optic=Optics(grid=grid,device='cpu')
    with torch.no_grad():m=optic.metrics(torch.zeros((1,40),dtype=torch.float64))
    fields=m['fields'][0].numpy()
    score=float(fields.mean()-.3*fields.std())
    assert abs(score-float(m['score'][0]))<1e-12
    row=dict(grid=grid,Q2000=score,MTF_mean=float(fields.mean()),MTF_std=float(fields.std()),
             MTF_min=float(fields.min()),fields=fields.tolist(),valid=bool(m['valid'][0]),
             wrms=m['wrms'][0].tolist(),support=m['support'][0].tolist(),seconds=time.perf_counter()-start)
    rows.append(row)
    print(json.dumps(row),flush=True)
result=dict(scope='Same optical model and scoring as exact11 one-shot evaluation; original prescription at q=0, no perturbation or control.',
            input_pose=[0.]*40,command=[0.]*40,training=0,inference=0,camera_acquisitions=0,
            source=str(source/'optics.py'),prescription=str(source/'prescription.json'),
            prescription_sha256=hashlib.sha256((source/'prescription.json').read_bytes()).hexdigest(),
            rows=rows,main_grid=129,refinement_grid=257,
            historical_note='Archived0.5962 is a plain nominal MTF mean from the older implementation, not this Q2000.')
(HERE/'nominal_reference_20261031.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
