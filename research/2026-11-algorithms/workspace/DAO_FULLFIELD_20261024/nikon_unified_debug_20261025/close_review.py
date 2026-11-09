"""Close the completed four-epoch and directional checks with bounded claims."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil
HERE=Path(__file__).resolve().parent
g=json.loads((HERE/'gradient_check/report.json').read_text())
checks=[]
for checkpoint,row in g['checkpoints'].items():
    assert abs(row['loss']-row['zero_step_loss'])<1e-7
    for direction,values in row['directions'].items():
        for value in values:
            assert value['actual_central_derivative']>0 and value['predicted_derivative']>0
            assert value['negative_gradient_change']<0
            checks.append(dict(checkpoint=checkpoint,direction=direction,**value))
assert len(checks)==18
review=dict(updated=datetime.now(timezone.utc).isoformat(),status='reviewed',
    training_cases=64,checkpoints=['initial','last'],central_difference_checks=18,optimizer_updates=0,
    checked_descent_directions_decrease_loss=18,sign_reversals_observed=0,
    total_direction_relative_amplitude_discrepancy_at_h1e4={
        name:row['directions']['total'][-1]['derivative_ratio']-1 for name,row in g['checkpoints'].items()},
    conclusion='No sign reversal in these sampled training directions; the approximate backward has magnitude errors, and this limited check does not establish its complete correctness or explain absent validation gain.',
    invalid_first_diagnostic_preserved='report_stale_direct_probe.json',
    scope='64 training samples only; no additional neural updates; no validation-based gradient tuning')
(HERE/'gradient_check/assistant_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
state=json.loads((HERE/'gradient_check/state.json').read_text());state['assistant_review']='reviewed'
(HERE/'gradient_check/state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
four=json.loads((HERE/'assistant_review.json').read_text())
four['ongoing_work']=None;four['gradient_check_review']=review
(HERE/'assistant_review.json').write_text(json.dumps(four,indent=2)+'\n',encoding='utf-8')
result=HERE/'RESULTS.txt'
if result.exists() and not (HERE/'PROGRAM_RESULTS.txt').exists():shutil.copyfile(result,HERE/'PROGRAM_RESULTS.txt')
lines=['712万统一Nikon网络：4轮训练及后续梯度检查完成，助手已核验。',
    '训练：752次更新，全部101个参数张量变化，7,121,987个FP32参数。',
    '口径：一次采集，4步模型内修正，无实测闭环，推理不读取真值；中心ABS/边缘DIFF，三场均<0.07 waves。',
    '原6000例：初始化1491/6000=24.85%；第1轮1470=24.50%；第2轮1488=24.80%；第3轮1485=24.75%；第4轮1492=24.8667%。',
    '第4轮相对初始新增97例通过、丢失96例，只净增1例，不能据此判断有实质提升。',
    '全7体运动：初始化106/1333=7.95%，末轮102/1333=7.65%；平均全场误差0.172688→0.179476 waves。',
    '同6000位姿换噪声：末轮1460/6000=24.33%；全7体96/1333=7.20%。不是独立新场景。',
    '78个更新监测点的最大预测变化RMS0.01949，最后0.001365；此前严重失稳没有重现。',
    '继续核对64训练样本、初始与末轮两检查点，共18组中心差分：所查负梯度方向均使实际损失下降，未见方向反号。',
    '总梯度方向的导数幅值在h=1e-4时偏差约19.5%和32.1%；不能称近似反传完全正确，亦不能据此解释无验证收益。',
    '本轮结果：训练稳定性修复成立，尚未证明恢复率学习提升；原28.77%分组参考模型继续保留，Trepan保持。',
    '当前本轮计算与核验均已结束，无新的训练后台。']
result.write_text('\n'.join(lines)+'\n',encoding='utf-8')
current='CURRENT '+review['updated']+'\n'+'\n'.join(lines)+'\n\n'
for path in (HERE/'HANDOFF.txt',HERE.parent/'HANDOFF.txt',HERE.parent/'nikon_short_training/HANDOFF.txt'):
    old=path.read_text(encoding='utf-8') if path.exists() else ''
    path.write_text(current+'--- Preserved previous handoff ---\n'+old,encoding='utf-8')
print(json.dumps(review,indent=2))
