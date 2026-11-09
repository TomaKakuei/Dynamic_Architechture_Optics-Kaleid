"""Audit saved scalar decisions and the Trepan validity decomposition."""
from pathlib import Path
import sys,os,json
ENV=Path(sys.prefix)
sys.path.append(str(ENV/'Lib/site-packages'))
_dll=os.add_dll_directory(str(ENV/'Library/bin'))
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'Adaptive Optics/ADAPTIVE_OPTICS_V15P'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
protocol=read(HERE/'protocol.json')
audit={}
for name in ('trepan','nikon','kla'):
    paths=[HERE/name/'cases'/f'{s}.json' for s in protocol['seeds'][name]]
    if not all(p.exists() for p in paths):continue
    rows=[read(p) for p in paths]
    if name=='nikon':
        archived=read(SOURCE/'kaleid_scope_repair_20260906/results/physical_check.json')['cases']
        lookup={r['seed']:r['hidden'] for r in archived}
        assert all(np.array_equal(r['initial'],lookup[r['seed']]) for r in rows)
    fields=np.array([r['fields'] for r in rows]);valid=np.array([r['valid'] for r in rows])
    if name=='kla':
        score=fields.mean(1)-.3*fields.std(1)
        assert np.allclose(score,[r['Q2000'] for r in rows],atol=1e-12,rtol=1e-12)
        optical=score>=.45
        all_fields=(fields>=.45).all(1)
    else:
        optical=(fields<.07).all(1);all_fields=optical
    assert np.array_equal(valid&optical,[r['pass'] for r in rows])
    audit[name]=dict(rows=len(rows),registered_passes=int((valid&optical).sum()),
        optical_only_registered_passes=int(optical.sum()),optical_all_positions=int(all_fields.sum()),
        valid_all_positions=int((valid&all_fields).sum()),invalid=[r['seed'] for r in rows if not r['valid']],
        mean_fields=fields.mean(0).tolist())
    audit[name]['finite_fields']=bool(np.isfinite(fields).all())
    audit[name]['exact_paper_seed_set']=sorted(r['seed'] for r in rows)==protocol['seeds'][name]
    if name=='kla':
        audit[name]['Q2000']=score.tolist()
        audit[name]['minimum_MTF']=fields.min(1).tolist()
sys.path.insert(0,str(SOURCE/'trepan_restored_joint_20261008'))
from geometry import Geometry
g=Geometry()
detail=[]
for p in sorted((HERE/'trepan/cases').glob('*.json')):
    r=read(p);q=np.array(r['initial']);ep=np.array(r['endpoint'])
    detail.append(dict(seed=r['seed'],initial_geometry=g.state(q),endpoint_geometry=g.state(ep),
        full_path=g.path(ep,q),minimum_initial_bound_mm=float(g.bounds(q).min()),
        minimum_endpoint_bound_mm=float(g.bounds(ep).min())))
audit['trepan_geometry']=detail
(HERE/'saved_result_checks.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print(json.dumps(audit,indent=2))
