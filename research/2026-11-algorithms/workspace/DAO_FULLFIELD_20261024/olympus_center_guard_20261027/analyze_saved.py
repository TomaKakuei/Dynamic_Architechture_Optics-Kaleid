"""CPU-only failure attribution from committed full-evaluation records."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,datetime
import numpy as np
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parent;FULL=WORK/'olympus_full500_20261027'
SOURCE=WORK.parent/'RAYOPTICS_RESTORE_20260929/dao_mod03/sampling_front3_rear4/olympus_joint_20261015'
KROOT=WORK/'olympus_constrained_solver_20261026';DROOT=WORK/'olympus_lowcost_refiner_20261026'
with np.load(SOURCE/'training/selected_validation.npz') as z:
    truth={int(i):q for i,q in zip(z['index'],z['initial'])}
G=np.load(DROOT/'center_gram.npy')
rows=[json.loads(p.read_text()) for p in sorted((FULL/'cases').glob('case_*.json'))]


def full(m):return m['valid'] and all(v is not None and v<.07 for v in m['fields'])
def center(m):return m['valid'] and m['fields'][0] is not None and m['fields'][0]<.07
def number(v):return float(v) if v is not None and np.isfinite(v) else None
def stage_details(r,name):
    if r['runtime']['reused_archived_case']:
        source=Path(r['provenance']['source'])
        if name=='N':return json.loads(source.read_text())
        tag='development_k_adaptive8' if source.parent.name.startswith('development') else 'paired32_k_adaptive8'
        return json.loads((KROOT/tag/f"case_{r['case']}.json").read_text())
    return json.loads((FULL/'stages'/name/f"case_{r['case']}.json").read_text())['details']


groups=dict(passed=[],invalid=[],center_failure=[],edge_only_failure=[])
lost=[];gained=[];center_losses=[];details=[];exceedances=[];invalid_transitions=[]
for r in rows:
    case=r['case'];metrics=r['metrics'];a,b=metrics['network'],metrics['N']
    if full(b):groups['passed'].append(case)
    elif not b['valid']:groups['invalid'].append(case)
    elif not center(b):groups['center_failure'].append(case)
    else:groups['edge_only_failure'].append(case)
    if full(a) and not full(b):lost.append(case)
    if not full(a) and full(b):gained.append(case)
    pose_rms={route:float(np.sqrt(np.mean((truth[case]+np.asarray(command))**2))) for route,command in r['commands'].items()}
    for before,after in (('network','D'),('D','K'),('K','N'),('network','N')):
        if metrics[before]['valid'] and not metrics[after]['valid']:
            endpoint=truth[case]+np.asarray(r['commands'][after]);axis=int(np.argmax(abs(endpoint)))
            invalid_transitions.append(dict(case=case,transition=before+'_to_'+after,
                endpoint_absmax=float(abs(endpoint).max()),outside_original_pose_domain=bool(abs(endpoint).max()>1),
                axis=axis,original_initial_coordinate=float(truth[case][axis]),
                command=float(r['commands'][after][axis]),neural_command=float(r['commands']['network'][axis]),
                note='Offline diagnosis only; unknown initial pose is not used by inference'))
    change=np.asarray(r['commands']['N'])-np.asarray(r['commands']['network'])
    proxy=float(np.sqrt(max(change@G@change,0)))
    physical_change=abs(b['fields'][0]-a['fields'][0]) if a['fields'][0] is not None and b['fields'][0] is not None else None
    if a['valid'] and b['valid'] and physical_change is not None and physical_change>proxy+1e-6:
        exceedances.append(dict(case=case,actual_abs_rms_change=physical_change,nominal_phase_change_proxy=proxy,
                                excess=physical_change-proxy,center_initial=a['fields'][0],center_final=b['fields'][0]))
    if center(a) and not center(b):
        first=next(name for name in ('D','K','N') if not center(metrics[name]))
        center_losses.append(dict(case=case,first_loss_stage=first,
            center_by_stage={name:metrics[name]['fields'][0] for name in metrics},
            valid_by_stage={name:metrics[name]['valid'] for name in metrics},pose_error_rms=pose_rms,
            nominal_phase_change_proxy=proxy))
    if not full(b):
        k,n=stage_details(r,'K'),stage_details(r,'N')
        cost=n.get('final_image_cost',k.get('final_image_cost'))
        steps=[s for s in n.get('steps',[]) if 'solver' in s]
        details.append(dict(case=case,valid=b['valid'],center=b['fields'][0],worst=number(max(v for v in b['fields'] if v is not None)) if any(v is not None for v in b['fields']) else None,
            final_image_merit=number(cost),image_merit_at_or_below9=bool(cost is not None and cost<=9),
            last_N_accepted=steps[-1]['accepted'] if steps else None,
            last_N_predicted_gain=steps[-1]['solver']['predicted_gain'] if steps else None,
            pose_error_rms=pose_rms,nominal_phase_change_proxy=proxy,actual_abs_center_rms_change=physical_change))
result=dict(updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=len(rows),total=500,
    complete=len(rows)==500 and (FULL/'result.json').exists(),groups=groups,fullfield_gained=gained,fullfield_lost=lost,
    center_losses=center_losses,remaining_failures=details,invalid_transitions=invalid_transitions,
    nominal_proxy_exceedances=sorted(exceedances,key=lambda r:-r['excess']),
    scope='Post-hoc diagnostic from saved commands/scores and labels; zero neural/optical inference, zero new scoring; not used by current evaluator',
    caution='Nominal phase-change proxy is an approximation, not a certificate at unknown residual poses. RMS-change comparison only diagnoses this approximation.')
for name in ('latest_diagnosis.json',f'diagnosis_{len(rows):03d}.json'):
    (ROOT/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(dict(completed=len(rows),complete=result['complete'],groups={k:len(v) for k,v in groups.items()},
    fullfield_lost=lost,center_loss_ids=[r['case'] for r in center_losses],
    largest_nominal_proxy_exceedances=result['nominal_proxy_exceedances'][:5]),indent=2))
