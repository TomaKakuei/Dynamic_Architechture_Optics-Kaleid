"""CPU-only saved-response analysis; no rendering, inference or scoring replay."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent
KROOT=WORK/'olympus_constrained_solver_20261026';DROOT=WORK/'olympus_lowcost_refiner_20261026'
FULL=WORK/'olympus_full500_20261027'
SOURCE=WORK.parent/'RAYOPTICS_RESTORE_20260929/dao_mod03/sampling_front3_rear4/olympus_joint_20261015'
sys.path.insert(0,str(KROOT))
from qp import solve_local
selection=json.loads((DROOT/'selection.json').read_text());center_gram=np.load(DROOT/'center_gram.npy')
with np.load(DROOT/'calibration.npz') as z:precision=np.linalg.inv(z['covariance'])/128**2
with np.load(WORK/'olympus_repair_4ep_20261026/evaluation500/predictions.npz') as z:
    priors={int(i):-q for i,q in zip(z['index'],z['command'])}
results=[]
for case in (7009,7034,7288,7354,7463,7488):
    dev=case in selection['development'];paired=case in selection['confirmation']
    tag='development_k_adaptive8' if dev else 'paired32_k_adaptive8' if paired else 'evaluation500_k_adaptive8'
    record=json.loads(((KROOT/tag/f'case_{case}.json') if dev or paired else (FULL/'stages/K'/f'case_{case}.json')).read_text())
    details=record if dev or paired else record['details']
    with np.load(SOURCE/'data'/f'batch_{case//25:04d}.npz') as z:obs=z['speckles'][case%25].astype(float)
    signal=np.maximum(obs-108,0);sigma=np.sqrt(4*signal+16*(1.2**2+.5**2)+(.005*signal)**2+1/12)
    for step in details['steps']:
        if 'solver' not in step:continue
        cycle=step['cycle'];path=KROOT/'responses'/tag/f'{case}_{cycle}.npz'
        if not path.exists():path=KROOT/'responses/development_j_central8'/f'{case}_{cycle}.npz'
        if not path.exists():continue
        with np.load(path) as z:
            x=z['prior'].copy();image=z['image'].astype(float);B=z['basis'][:,:8].copy()
            J=np.stack([z[f'jac_{k}'].astype(float) for k in range(8)],axis=-1)/sigma[...,None]/128
        J=J.reshape(9,-1,8);res=((image-obs)/sigma/128).reshape(9,-1)
        P=B.T@precision@B;C=B.T@center_gram@B;d=x-priors[case];prior_g=B.T@precision@d
        updates={};solver_info={}
        for name,js,rs in [('center',J[:1].reshape(-1,8),res[:1].ravel()),
                           ('edges',J[1:].reshape(-1,8),res[1:].ravel()),
                           ('all',J.reshape(-1,8),res.ravel())]:
            Q=js.T@js+P;g=js.T@rs+prior_g
            u,info=solve_local(Q,g,B,x,priors[case],center_gram,.2,step['center_limit'])
            updates[name]=u;solver_info[name]=info
        def cosine(a,b):
            u,v=updates[a],updates[b];den=np.sqrt(max(u@C@u,0)*max(v@C@v,0))
            return float(u@C@v/den) if den>1e-12 else None
        row=dict(case=case,cycle=cycle,center_limit=step['center_limit'],
            center_edge_cosine=cosine('center','edges'),center_all_cosine=cosine('center','all'),
            center_step_waves={name:float(np.sqrt(max(u@C@u,0))) for name,u in updates.items()},
            local_solver_success={name:info['success'] for name,info in solver_info.items()},
            cost_per_field=(res**2).sum(1).tolist())
        results.append(row)
(ROOT/'view_agreement.json').write_text(json.dumps(dict(scope='Saved-case CPU-only targeted diagnostic; no new physical endpoints',rows=results),indent=2)+'\n',encoding='utf-8')
for r in results:print(json.dumps({k:r[k] for k in ('case','cycle','center_edge_cosine','center_all_cosine')}))
