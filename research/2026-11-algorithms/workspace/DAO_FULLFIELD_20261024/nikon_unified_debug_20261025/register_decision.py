"""Register the diagnosed repair before its four-epoch training."""
from pathlib import Path
from datetime import datetime,timezone
import json
HERE=Path(__file__).resolve().parent
report=json.loads((HERE/'normalization_diagnosis.json').read_text())
name='normalized_backbone1e-7_readout1e-05'
probe=report['probes'][name]
assert probe['training_loss_after']<probe['training_loss_before']
assert probe['check_neural_prediction_delta_rms']<.025
assert probe['check_batch_after']['full_field']['field_mean']<probe['check_batch_before']['full_field']['field_mean']
assert probe['check_batch_after']['full_field']['full_field_passes']>=probe['check_batch_before']['full_field']['full_field_passes']
assert (HERE/'normalization_diagnostic_completed.json').exists()
decision=dict(passed=True,updated=datetime.now(timezone.utc).isoformat(),parameters=7121987,
    initialization_checkpoint='normalized_initialized.pt',
    initialization_description='Same training-only ridge fit as the 24.85% initial model, with feature centering/scaling explicit rather than folded into large weights; FP32 rounding differences measured',
    ridge=.001,feature_std_floor=1e-6,learning_rate=1e-7,readout_learning_rate=1e-5,
    selected_probe=name,selected_probe_result=probe,
    selection_reason='Highest tested readout learning rate that decreased the first training-batch loss and check-batch optical mean without reducing check-batch passes; all design checks use training samples',
    backbone_rate_reason='Original 1e-4 and 1e-6 destroyed the prior in one update; 1e-7 reduced training loss and retained check-batch recovery',
    diagnostic_discarded_optimizer_updates=16,actual_epochs_to_run=4,actual_updates_to_run=752,
    train_count=24000,validation_count=6000,full_network_trainable=True,
    measurement_acquisitions=1,model_only_refinements=4,measured_feedback=False,
    validation_used_for_repair_selection=False,fit=report['fit'],
    limitation='Single-step training diagnostics identify the failure and gate the trial; they do not establish four-epoch generalization')
path=HERE/'decision.json'
assert not path.exists(), 'Preserve the registered decision'
path.write_text(json.dumps(decision,indent=2)+'\n',encoding='utf-8')
text=('CURRENT '+decision['updated']+'\n'
    '用户要求debug并继续修复验证；新节点nikon_unified_debug_20261025。\n'
    '已复现原AdamW首步导致预测RMS变化19.05、训练检查样本均值0.175→78.21waves、42/128→0。\n'
    '根因：全位姿拟合折叠低方差特征后读出权重范数46.6万，仍沿用残差分支1e-4学习率；梯度有限不代表更新稳定。\n'
    '修复：把特征标准化显式留在网络，读出权重范数22.80；初始预测差RMS1.28e-5；主干lr1e-7、读出层lr1e-5。\n'
    '首步损失0.962655→0.958265，检查均值0.175325→0.174413，42/128保持；这不是完整训练结论。\n'
    '16次独立首步诊断更新全部丢弃；随后从normalized_initialized.pt真正训练4轮752次，全部7,121,987参数开放、FP32。\n'
    '一次采集、4步原模型修正、无实测闭环，中心ABS/边缘DIFF、全部<0.07waves；原24000/6000及1333全7体子集保持。\n'
    'state.json是总状态；stable_training/state.json是实际训练进度；run.log、stderr.log、update_diagnostics.jsonl保存日志。\n'
    '每10步记录实际网络输出变化，自动完成末轮、选中轮及同6000位姿新噪声检查；断点stable_training/resume.pt。\n'
    '预计约25–35分钟有效计算，按每批约1.6秒加全量验证估计；当前进程ID见started.json，先核对进程后续跑，勿重复启动。\n'
    '后台生成结果后仍需助手核验解释；没有主动通知配置。原28.77%模型与Trepan保持。\n')
(HERE/'HANDOFF.txt').write_text(text,encoding='utf-8')
for path in (HERE.parent/'HANDOFF.txt',HERE.parent/'nikon_short_training/HANDOFF.txt'):
    old=path.read_text(encoding='utf-8') if path.exists() else ''
    path.write_text(text+'\n--- Preserved previous handoff ---\n'+old,encoding='utf-8')
print(json.dumps(dict(registered=True,decision=str(HERE/'decision.json'))))
