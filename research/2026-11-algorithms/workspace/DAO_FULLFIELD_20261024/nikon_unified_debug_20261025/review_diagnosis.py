"""Record the verified failure and implemented repair, without a training claim."""
from pathlib import Path
from datetime import datetime,timezone
import json,numpy as np
HERE=Path(__file__).resolve().parent
previous=HERE.parent/'nikon_unified_repair_20261024'
d=json.loads((HERE/'diagnosis.json').read_text())
n=json.loads((HERE/'normalization_diagnosis.json').read_text())
decision=json.loads((HERE/'decision.json').read_text())
old=d['probes']['old_unbounded_0.0001']
new=n['probes'][decision['selected_probe']]
results=json.loads((previous/'comparison.json').read_text())
for epoch in range(5):
    with np.load(previous/'training'/('baseline_validation.npz' if epoch==0 else f'validation_{epoch:03d}.npz')) as z:
        p=z['valid'] & np.isfinite(z['fields']).all(1) & (z['fields'].max(1)<.07)
        assert int(p.sum())==results['original'][str(epoch)]['full_field_passes']
        assert len(p)==6000
review=dict(updated=datetime.now(timezone.utc).isoformat(),status='mechanism_reproduced_repair_training_started',
    failure=dict(readout_weight_norm=d['conditioning']['original_weight_norm'],
        near_constant_feature_count_below_001=d['conditioning']['features_below_1e3'],
        first_update_loss_before=old['training_loss_before'],first_update_loss_after=old['training_loss_after'],
        check_prediction_delta_rms=old['check_neural_prediction_delta_rms'],
        check_optical_mean_before=old['check_batch_before']['full_field']['field_mean'],
        check_optical_mean_after=old['check_batch_after']['full_field']['field_mean']),
    implemented=dict(explicit_training_feature_normalization=True,parameters=7121987,
        backbone_lr=decision['learning_rate'],readout_lr=decision['readout_learning_rate'],
        neural_parameters_dtype='float32',approximate_initial_prediction_equivalence=n['fit']['normalized_prior_difference'],
        readout_weight_norm=n['fit']['readout_weight_norm'],
        first_step_loss_before=new['training_loss_before'],first_step_loss_after=new['training_loss_after']),
    inference_contract=dict(one_acquisition=True,truth_input=False,measured_feedback=False,
        model_only_refinements=4,center='ABS',edges='DIFF',all_fields_threshold=.07),
    negative_controls_preserved=['Strong raw-feature ridge loses initial recovery',
        'Variance floor alone does not make the original learning rate stable',
        'LeakyReLU alone does not make the original learning rate stable'],
    prior_failed_run=dict(completed_epochs=4,updates=752,selected_epoch=0,
        initial_passes=results['original']['0']['full_field_passes'],
        final_passes=results['original']['4']['full_field_passes'],
        final_new_noise_passes=results['new_noise']['last']['full_field_passes']),
    scope='Training-sample paired first-step controls; not a guarantee of four-epoch validation improvement',
    missed_check='The preceding preflight checked finite gradients but omitted an actual optimizer-step effect check')
(HERE/'diagnosis_review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
(previous/'assistant_review.json').write_text(json.dumps(dict(updated=review['updated'],
    status='reviewed_not_adopted',completed_epochs=4,optimizer_updates=752,
    selected_epoch=0,initial_passes=1491,final_passes=351,final_rate=351/6000,
    final_noise_passes=347,diagnosis=str(HERE/'diagnosis_review.json'),
    next_run=str(HERE/'state.json')),indent=2)+'\n',encoding='utf-8')
lines=['Nikon四轮退化定位与修复（新四轮尚未完成）',
    '原4轮确实完成752次更新，全部101个参数张量变化；第4轮351/6000=5.85%，新噪声347/6000。',
    '不是只报告失败：已用同一初值和同一批训练数据复现第一次更新的失稳。',
    '原初始化把近常量GELU特征的逆方差折进最后线性层，权重范数46.6万；165/512个特征标准差低于0.001。',
    '原AdamW 1e-4首步损失0.962657→14.3394，训练检查样本预测变化RMS19.05，光学均值0.175315→78.2136waves，42/128→0。',
    '加大正则、只换激活均不能同时保持原预测能力和原学习率稳定性，负结果已保留。',
    '实现：显式保存训练特征中心/尺度，输出层权重范数22.80、偏置范数0.0826；原初始预测RMS差1.284e-5。',
    '主干学习率1e-7，最后线性层1e-5，所有7121987参数FP32开放。首步损失下降且检查样本42/128保持。',
    '16次独立诊断更新全部丢弃；新的4轮训练从初始化重新开始，752次更新，每10步记录实际输出变化。',
    '原6000例验证和同位姿新噪声测试将自动完成；一次采集、无真值输入、无实测闭环及中心ABS/边缘DIFF保持。',
    '这证明首步失稳机制及局部修复有效，不等于已经证明四轮训练泛化改善。']
(HERE/'DEBUG_REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(review=str(HERE/'diagnosis_review.json'),old_four_epoch_passes=351,new_four_epoch_status='running')))
