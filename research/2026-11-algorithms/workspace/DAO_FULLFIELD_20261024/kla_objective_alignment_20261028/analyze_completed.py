"""Read saved KLA endpoints to choose the next bounded training experiment."""
from pathlib import Path
import json, numpy as np
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'training_compare'
SOURCE=ROOT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
with np.load(RUN/'selected_validation3000_grid129.npz') as z:new={k:z[k].copy() for k in z.files}
with np.load(RUN/'selected_refined_scores.npz') as z:fine={k:z[k].copy() for k in z.files}
with np.load(SOURCE/'training/joint_validation3000.npz') as z:old={k:z[k].copy() for k in z.files}
with np.load(SOURCE/'training/grid129_values.npz') as z:oldfine={k:z[k].copy() for k in z.files}
original=read(SOURCE/'training/joint_result.json')
score=oldfine['score'].copy()
for r in original['grid257_borderline']:score[r['case']]=r['q257']
passed= fine['valid']&(fine['score']>=.45); oldpassed=oldfine['valid']&(score>=.45)
delta=new['command']-old['command'];mean=delta.mean(0)
cohorts={}
for c in range(3):
    mask=old['cohort']==c
    cohorts[str(c+2)]=dict(count=int(mask.sum()),old_pass=int(oldpassed[mask].sum()),new_pass=int(passed[mask].sum()),
        q_mean=float(fine['score'][mask].mean()),wrms_mean=float(new['wrms'][mask].mean()))
ranges={}
for low,high in ((-1,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.45),(.45,.5),(.5,1.)):
    mask=(fine['score']>=low)&(fine['score']<high)
    if not mask.any():continue
    wrms=new['wrms'][mask]
    ranges[f'{low}_{high}']=dict(count=int(mask.sum()),mean_wrms=float(wrms.mean()),max_field_wrms_median=float(np.median(wrms.max(1))),
        bounded_derivative_factor_median=float(np.median(1/(1+np.log1p((wrms/.07)**2))**2)),
        psf_error_mean=float(new['psf_hellinger'][mask].mean()))
rows={}
for name,mask in [('pass',passed),('fail',~passed),('lost',oldpassed&~passed),('gained',~oldpassed&passed)]:
    rows[name]=dict(count=int(mask.sum()),q_mean=float(fine['score'][mask].mean()),
        wrms_mean=float(new['wrms'][mask].mean()),field_wrms_mean=new['wrms'][mask].mean(0).tolist(),
        field_mtf_mean=new['fields'][mask].mean(0).tolist(),
        mean_pose_error_rms=float(np.sqrt(np.mean((old['initial'][mask]+new['command'][mask])**2))))
out=dict(status='saved_result_analysis_complete',count=3000,original_passes=int(oldpassed.sum()),new_passes=int(passed.sum()),
    gained=int((~oldpassed&passed).sum()),lost=int((oldpassed&~passed).sum()),new_invalid=int((~fine['valid']).sum()),
    cohorts=cohorts,score_bands=ranges,groups=rows,
    command_delta=dict(rms=float(np.sqrt(np.mean(delta**2))),common_mean_rms=float(np.sqrt(np.mean(mean**2))),
        common_bias_energy_fraction=float((mean**2).sum()/np.mean((delta**2).sum(1)))),
    scope='Saved results only; no new inference, endpoint replay, source changes or threshold changes')
(ROOT/'completed_analysis.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
