"""Describe saved outcomes only; no training, inference, or optical tracing."""
from pathlib import Path
import json, sys, os
sys.path.append(r'C:\Users\arthu\anaconda3\Lib\site-packages')
os.add_dll_directory(r'C:\Users\arthu\anaconda3\Library\bin')
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / 'DAO_FULLFIELD_20261024'
SOURCE = ROOT / 'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
spec = read(SOURCE / 'prescription.json')
with np.load(SOURCE / 'training/joint_validation3000.npz') as z:
    ids, cohorts, poses = z['indices'], z['cohort'], z['initial']

def kla(folder):
    rows=[]
    for p in sorted((folder / 'cases').glob('*.json')):
        rows.append(read(p))
        if len(rows)%50==0: print('Read saved KLA cases:',len(rows),flush=True)
    arms = ['one_shot', 'feedback1', 'feedback2']
    q = np.array([[r['scores'][a]['score'] for a in arms] for r in rows])
    valid = np.array([[r['scores'][a]['valid'] for a in arms] for r in rows])
    passes = (q >= .45) & valid
    assert all(ids[r['row']] == r['case'] for r in rows)
    c = np.array([cohorts[r['row']] for r in rows])
    initial = np.stack([poses[r['row']] for r in rows])
    fail = ~passes[:, 2]
    caps = np.array([[s['fixed_bound_scale'] for s in r['steps']] for r in rows])
    raw = np.array([[s['raw_max'] for s in r['steps']] for r in rows])
    if len(rows) == 150:
        assert passes.sum(0).tolist() == [82, 99, 113]
    stats = dict(n=len(rows), passes=passes.sum(0).tolist(), q_means=q.mean(0).tolist(),
        invalid=(~valid).sum(0).tolist(), final_failures=int(fail.sum()),
        q_improved_last_step=int((q[:,2] > q[:,1]).sum()),
        failed_q_improved_last_step=int(((q[:,2] > q[:,1]) & fail).sum()),
        failed_q_improved_since_one_shot=int(((q[:,2] > q[:,0]) & fail).sum()),
        failed_q_bins={str(lo)+'..'+str(hi):int((fail & (q[:,2]>=lo) & (q[:,2]<hi)).sum()) for lo,hi in [(0,.3),(.3,.4),(.4,.45)]},
        step_capped_counts=(caps < 1-1e-12).sum(0).tolist(),
        failed_step_capped_counts=(caps[fail] < 1-1e-12).sum(0).tolist(),
        scale_median=np.median(caps,axis=0).tolist(),
        failed_scale_median=np.median(caps[fail],axis=0).tolist(),
        raw_command_max_median=np.median(raw,axis=0).tolist(),
        cohorts={str(i):dict(n=int((c==i).sum()),passes=passes[c==i].sum(0).tolist(), q_means=q[c==i].mean(0).tolist()) for i in np.unique(c)},
        initial_pose_rms_median=dict(passed=float(np.median(np.sqrt((initial[~fail]**2).mean(1)))),failed=float(np.median(np.sqrt((initial[fail]**2).mean(1))))),
        acquisition_count=3, note='Saved-case descriptive analysis; no isolation of structural causes.')
    if (folder/'state.json').exists():
        state=read(folder/'state.json')
        stats['run_state']={k:state[k] for k in ['stage','error','updated'] if k in state}
    return stats

olympus={}
with np.load(WORK/'olympus_full500_20261027/predictions500.npz') as z:
    for arm in ['network','N']:
        fields=z['fields_'+arm]; valid=z['valid_'+arm]; below=fields<.07
        center=below[:,0]&valid
        inner=below[:,:5].all(1)&valid
        full=below.all(1)&valid
        olympus[arm]=dict(n=len(fields),center=int(center.sum()),center_and_inner15deg=int(inner.sum()),
            all9_including_outer3deg=int(full.sum()),outer_only_failure=int((inner & ~full).sum()),
            center_pass_but_inner_failure=int((center & ~inner).sum()),
            valid_center_failure=int((valid & ~below[:,0]).sum()),invalid=int((~valid).sum()),
            per_field_passes=(below & valid[:,None]).sum(0).tolist(),
            per_field_means=fields.mean(0).tolist())
    assert olympus['N']['all9_including_outer3deg']==373
diagnosis=read(WORK/'olympus_center_guard_20261027/diagnosis_500.json')
olympus['saved_failure_groups']={k:len(v) for k,v in diagnosis['groups'].items()}
olympus['failures_with_image_merit_at_or_below9']=sum(x['image_merit_at_or_below9'] for x in diagnosis['remaining_failures'])

print('Finished Olympus saved-array analysis.',flush=True)
result=dict(scope='Read-only analysis of saved predictions, commands, and source metadata. No new inference, training, optical scoring, or resumption.',
    kla_structure=dict(groups=spec['groups'], controlled_dof=len(spec['groups'])*5,
        mirror_surface_ids=[i for i,s in enumerate(spec['surfaces']) if s['mat']=='mirror'],
        controlled_mirror_surfaces=[i for i,s in enumerate(spec['surfaces']) if s['mat']=='mirror' and any(a<=i<=b for a,b in spec['groups'])],
        upstream_catadioptric_section='Fixed; source Optics initializes and caches prefix through surface14.',
        note='Three dominant-module cohorts share the same complete optical system and joint observation; not isolated modules.'),
    kla_complete150=kla(WORK/'kla_feedback_loop150_20261028'),
    kla_partial_extension=read(WORK/'kla_feedback_full3000_20261028/partial_summary.json')['metrics'],
    kla_extension_state=read(WORK/'kla_feedback_full3000_20261028/state.json'),olympus=olympus)
(HERE/'saved_analysis.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
