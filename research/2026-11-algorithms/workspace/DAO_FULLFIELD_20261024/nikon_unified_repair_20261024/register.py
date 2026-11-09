"""Register the authorized run and preserve older handoff entries."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
HERE=Path(__file__).resolve().parent
source=HERE.parent/'nikon_short_training/exact_local_response/candidate.pt'
files=[HERE/'repair.py',HERE/'prepare.py',HERE/'run.py',HERE/'initialized.pt',source]
record=dict(created=datetime.now(timezone.utc).isoformat(),parameters=7121987,
    authorized_epochs=4,optimizer_updates=752,train_count=24000,validation_count=6000,
    readout_initialization=dict(training_labels=24000,ridge=.001,extra_optimizer_epochs=0),
    direct_prior_loss=dict(weight=1.,kind='SmoothL1',beta=.25),
    selection='Original validation only; maximum full-field passes then worst P90 then field mean; E0 eligible',
    inference=dict(acquisitions=1,model_only_refinements=4,measured_feedback=False,truth_input=False),
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
path=HERE/'plan.json'
if not path.exists():path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
for path in (HERE.parent/'HANDOFF.txt',HERE.parent/'nikon_short_training/HANDOFF.txt'):
    old=path.read_text(encoding='utf-8') if path.exists() else ''
    marker='CURRENT REPAIR nikon_unified_repair_20261024'
    if not old.startswith(marker):
        current=(marker+' '+record['created']+'\n'
            '用户已授权：修好712万统一网络后再训练4轮。已启动新目录run.py，PID见该目录started.json。\n'
            '修复完整35维输出头的训练集初始化，并对原始网络输出加直接位姿监督；原4步模型修正及中心ABS/边缘DIFF保持。\n'
            '网络7,121,987参数、FP32；严格4轮752次更新，另有一次24000训练样本线性输出头拟合。\n'
            '最新状态/日志/断点/结果均见nikon_unified_repair_20261024/HANDOFF.txt、state.json、training/state.json、run.log。\n'
            '原候选与Trepan保持；不要重复启动旧任务。\n\n--- Preserved previous handoff ---\n')
        path.write_text(current+old,encoding='utf-8')

if __name__=='__main__':print(json.dumps(dict(plan=str(HERE/'plan.json'),registered=True)))
