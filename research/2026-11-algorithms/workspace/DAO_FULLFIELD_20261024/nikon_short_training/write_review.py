"""Collect completed experiments without rerunning training or optical scoring."""
from pathlib import Path
import csv,json
from datetime import datetime,timezone
import numpy as np

HERE=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def stamp():return datetime.now(timezone.utc).isoformat()

assert read(HERE/'state.json')['status']=='completed'
assert read(HERE/'exact_completion_state.json')['status']=='completed'
check=read(HERE/'inference_and_parameter_check.json');assert check['passed']
comparison=read(HERE/'comparison.json')['results']
rows=[]
for idea,r in comparison.items():
    evidence=r['training'];assert evidence['completed_epochs']==4 and evidence['optimizer_updates']==752
    rows.append(dict(idea=idea,epochs=4,optimizer_updates=752,selected_epoch=r['selected_epoch'],
        baseline_passes=r['baseline']['full_field_passes'],epoch4_passes=r['epoch4']['full_field_passes'],
        selected_passes=r['best']['full_field_passes'],count=r['best']['count'],protocol=r['inference_protocol'],
        active_seconds=r['active_seconds']))
with (HERE/'all_short_trials.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)

protocols={};array_checks=[]
for idea in ('closed_loop_training','nonlinear_local_model','exact_local_response'):
    protocols[idea]={}
    for mode,dirname in [('original','original_protocol_comparison'),('new_noise','noise_confirmation')]:
        folder=HERE/idea/dirname
        if not (folder/'summary.json').exists():continue
        report=read(folder/'summary.json');protocols[idea][mode]={}
        for name,r in report['results'].items():
            with np.load(folder/(name+'.npz')) as z:
                n=len(z['fields']);valid=z['valid'].astype(bool)
                passes=valid & np.all(z['fields']<.07,axis=1)
                counts={str(int(g)):int(((z['cohort']==g)&passes).sum()) for g in np.unique(z['cohort'])}
                assert n==6000 and len(np.unique(z['original_ids']))==6000
                assert int(passes.sum())==r['validation']['full_field_passes']
                assert int((~valid).sum())==r['validation']['invalid']
                assert all(int((z['cohort']==g).sum())==2000 for g in np.unique(z['cohort']))
                assert np.isfinite(z['command']).all() and np.max(np.abs(z['command']))<=1.500001
                # Reported fields must use center ABS, not outer-field ABS.
                assert np.allclose(z['fields'][:,0],z['absolute'][:,0],rtol=0,atol=1e-12)
            protocols[idea][mode][name]=dict(passes=int(passes.sum()),count=n,rate=float(passes.mean()),
                invalid=int((~valid).sum()),cohort_passes=counts,field_mean=r['validation']['field_mean'],
                acquisitions_mean=r['acquisitions_mean'],epoch=r['epoch'],checkpoint=r['checkpoint'])
            array_checks.append(dict(idea=idea,mode=mode,route=name,passed=True))

exact=protocols['exact_local_response'];selected=read(HERE/'exact_local_response/summary.json')
assert selected['selected_epoch']==0
historical11=dict(original=read(HERE.parent/'nikon_debug_20261024/closed_loop_summary.json')['methods'],
    exact_epoch4=read(HERE/'exact_local_response/loop_summary.json'))
recommendation=dict(idea='exact_local_response',checkpoint=str(HERE/'exact_local_response/candidate.pt'),
    selected_epoch=0,network_source=str(HERE/'nonlinear_local_model/candidate.pt'),network_source_epoch=4,
    inference_protocol='One acquisition: FP32 network plus4model-only exact-response refinements',
    training_attribution='This trial4epoch fine-tuning rejected for one-acquisition inference; epoch4 retained separately for measured-loop comparison. Selected one-acquisition weights come from previous successful4epoch local-model trial',
    original=exact['original']['idea_selected'],new_noise=exact['new_noise']['idea_selected'],
    same_checkpoint_with_original_measured_loop=dict(original=exact['original']['idea_selected_measured_loop'],
        new_noise=exact['new_noise']['idea_selected_measured_loop']),
    separate_epoch4_closed_loop_candidate=dict(checkpoint=str(HERE/'exact_local_response/last.pt'),
        original=exact['original']['idea_epoch4_measured_loop'],new_noise=exact['new_noise']['idea_epoch4_measured_loop'],
        one_acquisition_original=exact['original']['idea_epoch4'],one_acquisition_new_noise=exact['new_noise']['idea_epoch4']),
    default_entry=str(HERE/'predict_nikon.py'),original_default_overwritten=False,
    metric='Center ABS and both outer-field DIFF relative to fixed corrected bias; all3<0.07waves',
    new_noise_independent_scenes=False,updated=stamp())
write(HERE/'recommended_nikon_model.json',recommendation)
review=dict(status='completed_and_assistant_checked',updated=stamp(),trials=rows,protocols=protocols,historical11=historical11,
    array_checks=array_checks,inference_check=check,recommendation=recommendation,
    metric='center ABS; edge DIFF; evaluation fields0,+/-0.1mm,197pupil nodes; all3<0.07waves',
    data='Original24000training/6000validation;3cohorts with2000validation each',
    new_noise_scope='Same6000poses, separate fixed measurement noise; not independent scenes',
    training_dtype='All7121929neural parameters FP32; precise optical response/scoring FP64',
    trepan='Unchanged; prior training remains stopped',
    original_default_overwritten=False,
    diagnosis=['Center correction alone did not control outer-field DIFF.',
        'Fixed nominal inverse severely amplified nonlinear model error; limiting its step did not solve the full-field problem.',
        'State-dependent nonlinear refinement and training through actual measured feedback both produced substantial recovery.',
        'Pose weighting and sensor consistency alone failed; exact-response implicit-gradient fine-tuning degraded one-acquisition recovery while improving original-noise measured-loop pass count.',
        'Remaining failures and incomplete observability are unresolved; no complete-cure or independent-scene claim.'])
write(HERE/'assistant_review.json',review)
lines=['Nikon四轮短训练：已完成并核对',f'更新时间UTC：{review["updated"]}',
    f'累计{len(rows)}项想法，每项实训4轮、752次更新；原24000训练/6000验证不变。',
    '口径：中心ABS、两侧边缘DIFF；三场同时小于0.07 waves才成功。没有剔除失败样本。',
    '网络7121929参数全部FP32。光学精确响应和评分仍为FP64。Trepan保持原状态。','',
    '全6000案例结果：原观测 / 同一批位姿更换测量噪声（不是独立场景）']
for idea,label in [('closed_loop_training','闭环端到端训练'),('nonlinear_local_model','单次采集+近似局部响应'),('exact_local_response','单次采集+精确局部响应')]:
    report=protocols[idea]
    original=report.get('original',{}).get('idea_selected')
    if original is None:
        original=dict(passes=comparison[idea]['best']['full_field_passes'],count=6000,rate=comparison[idea]['best']['full_field_rate'])
    new=report['new_noise']['idea_selected']
    lines.append(f'{label}：{original["passes"]}/6000 ({original["rate"]:.2%}) / {new["passes"]}/6000 ({new["rate"]:.2%})')
    if 'idea_selected_measured_loop' in report.get('original',{}):
        o=report['original']['idea_selected_measured_loop'];n=report['new_noise']['idea_selected_measured_loop']
        lines.append(f'  同一检查点再接原闭环：{o["passes"]}/6000 ({o["rate"]:.2%}) / {n["passes"]}/6000 ({n["rate"]:.2%})；原噪声平均读取{o["acquisitions_mean"]:.3f}次')
    if 'idea_epoch4_measured_loop' in report.get('original',{}):
        o=report['original']['idea_epoch4_measured_loop'];n=report['new_noise']['idea_epoch4_measured_loop']
        lines.append(f'  另存第4轮权重(last.pt)接原闭环：{o["passes"]}/6000 ({o["rate"]:.2%}) / {n["passes"]}/6000 ({n["rate"]:.2%})；原噪声平均读取{o["acquisitions_mean"]:.3f}次')
lines.extend(['','训练归因：',
    '闭环路线：原权重加闭环46.70%，再实训4轮54.85%；纯网络直接输出仍0%。',
    '近似局部响应：添加修正层14.37%，再实训4轮19.17%。',
    '精确局部响应：以上一项已训练权重为起点，换响应层28.77%；新4轮单次采集退步到24.85%，单次采集保留本轮epoch0。',
    '接回原闭环后，该项新4轮在原噪声下由60.73%升至61.88%；与单次采集结果分别报告，last.pt保留为闭环候选。',
    '不能把精确响应层收益说成这次微调收益，也不能把单次采集加模型求解称为纯网络一次前向。','',
    '原11例回归：精确响应第4轮单次采集6/11、闭环8/11；原权重校准闭环9/11，存在逐例退步。',
    '该11例使用last.pt；没有将其结果冒充单次采集所选epoch0检查点结果。','',
    '病因证据：固定名义线性逆在远离偏置处失真并放大误差；原训练优先修正中心，边缘DIFF没有同步恢复。',
    '改损失/位姿权重不足以解决；随当前状态更新的非线性响应和实测闭环有效。剩余失败尚未全部解释或消除。','',
    '使用入口：predict_nikon.py --input 输入.npz --output 指令.npz',
    '输入仅raw[N,3,49]和可选maps[N,3,32,32]；无位姿或真值标签。默认读取本目录recommended_nikon_model.json。',
    '输出35维归一化执行器指令；该入口执行单次采集内的4次模型修正，不执行实测闭环。',
    f'128条原观测CLI与保存指令最大差：{check["max_command_difference"]:.9g}。',
    '候选和全部失败结果另存；未覆盖原模型、未恢复旧长训队列。',
    '完整逐想法表：all_short_trials.csv；完整证据和各组失败数：assistant_review.json。'])
(HERE/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
current='CURRENT '+stamp()+'\n四轮短训练及全量确认已完成，助手已核对。无待续训练。\n详见RESULTS.txt和assistant_review.json。\n默认研究推理候选：exact_local_response/candidate.pt（本轮epoch0，源自上一项已训练epoch4）。\nTrepan和原长训队列仍停止；未覆盖原正式权重。\n\n'
for path in (HERE/'HANDOFF.txt',HERE.parent/'HANDOFF.txt'):
    old=path.read_text(encoding='utf-8-sig');path.write_text(current+'--- Preserved previous handoff ---\n'+old,encoding='utf-8')
print('\n'.join(lines))
