"""Check completed saved results and document the initialization limitation."""
from pathlib import Path
import os,json,shutil
from datetime import datetime,timezone
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import torch
torch.set_num_threads(1)
HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def counts(path,expected_ids=None):
    with np.load(path) as z:
        assert len(z['fields'])==6000 and len(np.unique(z['original_ids']))==6000
        if expected_ids is not None:assert np.array_equal(z['original_ids'],expected_ids)
        assert np.isfinite(z['command']).all() and np.max(np.abs(z['command']))<=1.500001
        assert np.allclose(z['fields'][:,0],z['absolute'][:,0],atol=1e-12,rtol=0)
        active=np.any(z['initial'].reshape(-1,7,5)!=0,axis=2).sum(1)
        seven=active==7;assert seven.sum()==1333
        passing=z['valid']&np.all(z['fields']<.07,axis=1)
        return dict(count=6000,passes=int(passing.sum()),all7_count=1333,
            all7_passes=int(passing[seven].sum()),invalid=int((~z['valid']).sum()))

assert read(HERE/'state.json')['status']=='completed'
comparison=read(HERE/'comparison.json');rows={};checked=0
with np.load(HERE.parent/'nikon_short_training/exact_local_response/selected_validation.npz') as z:
    original_ids=z['original_ids'].copy()
noise_ids=(original_ids.astype(np.uint64)+1000000007)%(2**32)
for size in ('1x','2x'):
    folder=HERE/('unified_'+size);source=comparison['models'][size]
    history=read(folder/'history.json')
    assert [r['epoch'] for r in history]==list(range(5))
    epochs={}
    for epoch in range(5):
        name='baseline_validation.npz' if epoch==0 else f'validation_{epoch:03d}.npz'
        row=counts(folder/name,original_ids);epochs[str(epoch)]=row
        assert row['passes']==source['original'][str(epoch)]['full_field_passes']
        assert row['all7_passes']==source['original'][str(epoch)]['all7']['full_field_passes']
        checked+=1
    best=counts(folder/'selected_validation.npz',original_ids)
    assert best==epochs[str(source['selected_epoch'])];checked+=1
    noisy={}
    for checkpoint in ('candidate','last'):
        noisy[checkpoint]=counts(folder/('new_noise_'+checkpoint+'.npz'),noise_ids)
        assert noisy[checkpoint]['passes']==source['new_noise'][checkpoint]['full_field_passes'];checked+=1
    resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
    state=resume['optimizer']['state'];steps=sorted({int(r['step']) for r in state.values()})
    assert steps==[752] and len(state)==101
    parameter_count=sum(r['exp_avg'].numel() for r in state.values())
    assert parameter_count==source['training']['trainable_parameters']
    assert all(r['exp_avg'].dtype==torch.float32 for r in state.values())
    rows[size]=dict(parameters=parameter_count,optimizer_updates=steps[0],parameter_tensors=len(state),
        epochs=epochs,selected_epoch=source['selected_epoch'],selected=best,new_noise=noisy,
        total_registered_active_seconds=sum(r['active_seconds'] for r in history))
    del resume,state
now=datetime.now(timezone.utc).isoformat()
review=dict(status='reviewed_not_adopted',updated=now,models=rows,array_sets_checked=checked,
    metric='center ABS, both outer-field DIFF; all3<0.07waves',
    inference='One observation packet,4model-only refinements, no measured feedback, no ground truth input',
    old_reference=dict(full_passes=1726,count=6000,all7_passes=159,all7_count=1333),
    conclusion='Both requested4epoch trials failed to recover old performance; doubling FFN capacity did not improve this initialization and training setting.',
    limitation='New forward returns the transferred residual head alone. Old forward adds that head to the old grouped/coordinated predictor. Removing that base term changes prediction semantics: own epoch0 already falls from28.77percent to0.55percent. Thus old-versus-unified comparison confounds architecture with lost initialization, and cannot establish unified architectures are inferior.',
    preserved='Original candidate, negative experiments and Trepan unchanged; no automatic adoption',
    suggested_next_design='First train/initialize the unified35D predictor to represent the complete old predictor, using training-only labels or teacher predictions and explicitly accounting for that training budget; then compare capacity under matched conditions.',
    independent_scene_claim=False,network_dtype='float32')
write(HERE/'assistant_review.json',review)
lines=['统一网络大小对照：助手已核对，两个候选均不采用',f'核对时间UTC：{now}',
    '两模型均实训4轮、各752次更新；101个参数张量均实际变化，神经网络FP32。',
    '原24000训练/6000验证；一次采集+4步模型修正，无实测闭环；中心ABS/边缘DIFF，三场均<0.07waves。',
    '旧完整模型：混合集1726/6000=28.77%；全7组同动159/1333=11.93%。','']
for size,r in rows.items():
    best=r['selected'];last=r['epochs']['4'];noise=r['new_noise']['candidate']
    lines.extend([f'{size}：{r["parameters"]}参数，最佳epoch{r["selected_epoch"]}。',
        f'  最佳：混合{best["passes"]}/6000={best["passes"]/6000:.2%}；全7动{best["all7_passes"]}/1333。',
        f'  第4轮：混合{last["passes"]}/6000={last["passes"]/6000:.2%}；全7动{last["all7_passes"]}/1333。',
        f'  最佳权重换噪声：{noise["passes"]}/6000={noise["passes"]/6000:.2%}，全7动{noise["all7_passes"]}/1333。'])
lines.extend(['','解释与限制：',
    '本次只搬迁了旧模型的全局残差分支，移除了原分组/协调预测，把残差输出直接当成完整位姿预测。',
    '迁移没有保留旧完整模型的初始函数：两种新规模训练前已降到33/6000=0.55%，全场均值约41.26waves。',
    '四轮训练没有恢复这部分能力。容量翻倍在本设置下没有收益；不能据此宣称统一网络本身不可行。',
    '两个新规模之间初始函数基本一致；相对旧模型的比较同时改变结构和已有预测能力。',
    '扩容具体为四个Transformer前馈层宽度1648→5120，总参数约翻倍，CNN与层数保持。',
    '下一项设计应先让统一网络学会完整35维预测，明确计入初始化/预训练预算，再做规模比较。',
    '新噪声仍使用同一6000个位姿，不是独立场景验证。',
    f'已检查{checked}组保存预测及优化器步数；没有重训或重跑光学求解。',
    '当前无本轮后台任务；原候选、全部失败记录、Trepan保持。'])
if not (HERE/'PROGRAM_RESULTS.txt').exists():shutil.copyfile(HERE/'RESULTS.txt',HERE/'PROGRAM_RESULTS.txt')
(HERE/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for path in (HERE/'HANDOFF.txt',HERE.parent/'HANDOFF.txt',HERE.parent/'nikon_short_training/HANDOFF.txt'):
    old=path.read_text(encoding='utf-8-sig')
    top=f'CURRENT {now}\n统一网络两个规模均完成4轮，助手已核对；候选均不采用，无本轮后台任务。\n最佳混合恢复率1x=0.75%、2x=0.6833%；全7组同动均0%。初始迁移丢失旧完整预测能力，不能归因于统一架构本身。\n详情：DAO_FULLFIELD_20261024/nikon_unified_20261024/RESULTS.txt及assistant_review.json。原候选与Trepan未变。\n\n'
    path.write_text(top+'--- Preserved previous handoff ---\n'+old,encoding='utf-8')
state=read(HERE/'state.json');state.update(assistant_review='completed_not_adopted',assistant_review_time=now);write(HERE/'state.json',state)
comparison.update(assistant_review='completed_not_adopted',assistant_review_file='assistant_review.json');write(HERE/'comparison.json',comparison)
print(json.dumps(dict(status=review['status'],array_sets_checked=checked,selected={k:v['selected'] for k,v in rows.items()}),indent=2))
