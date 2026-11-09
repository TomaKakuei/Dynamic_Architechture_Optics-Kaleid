"""Verify the completed four-epoch results from saved per-case arrays."""
from pathlib import Path
from datetime import datetime,timezone
import json,numpy as np,torch
HERE=Path(__file__).resolve().parent
RUN=HERE/'stable_training'
report=json.loads((HERE/'comparison.json').read_text())
rows=[];previous=None;first=None;last=None
for epoch in range(5):
    with np.load(RUN/('baseline_validation.npz' if epoch==0 else f'validation_{epoch:03d}.npz')) as z:
        valid=z['valid'] & np.isfinite(z['fields']).all(1)
        passed=valid & (z['fields'].max(1)<.07)
        moving=(z['initial'].reshape(-1,7,5)!=0).any(2).sum(1)
        m=report['original'][str(epoch)]
        assert len(passed)==6000 and int(passed.sum())==m['full_field_passes']
        assert int((~valid).sum())==m['invalid']
        assert int((moving==7).sum())==1333 and int(passed[moving==7].sum())==m['all7']['full_field_passes']
        if previous is not None:assert np.array_equal(previous,z['original_ids'])
        previous=z['original_ids'].copy()
        if epoch==0:first=passed.copy()
        if epoch==4:last=passed.copy()
        rows.append(dict(epoch=epoch,passes=int(passed.sum()),rate=float(passed.mean()),
            all7_passes=int(passed[moving==7].sum()),all7_rate=float(passed[moving==7].mean()),
            invalid=int((~valid).sum()),field_mean=float(z['fields'][valid].mean())))
for name in ('candidate','last'):
    with np.load(RUN/('new_noise_'+name+'.npz')) as z:
        passed=z['valid'] & np.isfinite(z['fields']).all(1) & (z['fields'].max(1)<.07)
        assert len(passed)==6000 and int(passed.sum())==report['new_noise'][name]['full_field_passes']
saved=torch.load(RUN/'resume.pt',map_location='cpu',weights_only=False)
steps=sorted({int(s['step']) for s in saved['optimizer']['state'].values()})
assert steps==[752] and len(saved['optimizer']['state'])==101
parameters={k:v for k,v in saved['model'].items() if not k.startswith(('base.','native_optics.','feature_center','feature_scale'))}
assert sum(v.numel() for v in parameters.values())==7121987
assert all(v.dtype==torch.float32 for v in parameters.values())
updates=[json.loads(line) for line in (HERE/'update_diagnostics.jsonl').read_text().splitlines()]
review=dict(updated=datetime.now(timezone.utc).isoformat(),status='reviewed_not_adopted',
    completed_epochs=4,optimizer_updates=752,parameters=7121987,parameter_tensors=101,
    rows=rows,selected_epoch=report['selected_epoch'],new_noise_rate=report['new_noise']['last']['full_field_rate'],
    new_noise_passes=report['new_noise']['last']['full_field_passes'],
    new_passes_vs_initial=int((last & ~first).sum()),lost_passes_vs_initial=int((first & ~last).sum()),
    update_monitor_records=len(updates),max_observed_prediction_update_rms=max(x['neural_prediction_delta_rms'] for x in updates),
    last_observed_prediction_update_rms=updates[-1]['neural_prediction_delta_rms'],
    conclusion='Severe update instability did not recur. Final recovery exceeds initial by one case, while all7 and mean optical error worsen; no clear learning improvement demonstrated.',
    ongoing_work='Training-only directional check of the approximate implicit-GN backward, gradient_check/state.json',
    original_modular_model_retained=True,trepan_unchanged=True)
(HERE/'assistant_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
report['assistant_review']='reviewed_not_adopted'
(HERE/'comparison.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
state=json.loads((HERE/'state.json').read_text());state['assistant_review']='reviewed_not_adopted'
(HERE/'state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
current=(f'CURRENT {review["updated"]}\n'
    '4轮752次更新及新噪声复测全部完成并已核对；训练稳定，但未证明有学习提升，候选不采用。\n'
    '初始化1491/6000=24.85%；E1=1470，E2=1488，E3=1485，E4=1492/6000=24.8667%；新噪声1460/6000=24.3333%。\n'
    '全7体初始化106/1333，末轮102/1333，新噪声96/1333。平均光学误差0.172688→0.179476waves。\n'
    '当前继续训练集64例的4步模型修正反向梯度检查，无新训练更新；状态gradient_check/state.json，结果gradient_check/report.json。\n'
    '原始诊断第一次数值探针遗漏刷新直接位姿缓存，记录另存report_stale_direct_probe.json；已修正，仅影响该诊断，不影响4轮训练或验证。\n\n')
for path in (HERE/'HANDOFF.txt',HERE.parent/'HANDOFF.txt',HERE.parent/'nikon_short_training/HANDOFF.txt'):
    old=path.read_text(encoding='utf-8') if path.exists() else ''
    path.write_text(current+'--- Previous handoff ---\n'+old,encoding='utf-8')
print(json.dumps(review,indent=2))
