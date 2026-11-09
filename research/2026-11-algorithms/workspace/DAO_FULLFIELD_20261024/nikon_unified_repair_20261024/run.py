"""Resumable four-epoch repair followed by serial, complete validation."""
from repair import *
import argparse,csv,ctypes,hashlib,os,shutil,traceback
RUN=HERE/'training'

def state(status,**extra):
    exp.atomic(HERE/'state.json',dict(status=status,pid=os.getpid(),updated=exp.stamp(),**extra))

def saved_metrics(path):
    with np.load(path) as z:
        f,v,a=z['fields'],z['valid'],z['absolute']
        moving=np.any(z['initial'].reshape(-1,7,5)!=0,axis=2).sum(1)
        result=exp.aggregate(f,v,a,.07)
        result['by_moving_bodies']={str(k):exp.aggregate(f[moving==k],v[moving==k],a[moving==k],.07) for k in np.unique(moving)}
        result['all7']=result['by_moving_bodies']['7']
        assert result['count']==6000 and result['all7']['count']==1333
        assert np.isfinite(z['command']).all()
    return result

def training():
    state('training',details=str(RUN/'state.json'))
    system=RepairSystem()
    args=argparse.Namespace(stage='train',system='nikon',variant='large',epochs=4,batch=128,microbatch=128,seed=SEED)
    exp.train(system,args,out=RUN)
    last=torch.load(RUN/'last.pt',map_location='cpu',weights_only=False)
    resume=torch.load(RUN/'resume.pt',map_location='cpu',weights_only=False)
    initial=torch.load(HERE/'initialized.pt',map_location='cpu',weights_only=False)['model']
    model=system.model('large')
    differences={k:float((last['model'][k].double()-initial[k].cpu().double()).square().sum()) for k,_ in model.named_parameters()}
    steps=sorted({int(v['step']) for v in resume['optimizer']['state'].values()})
    assert steps==[752] and len(resume['history'])==5 and last['epoch']==4
    assert all(v>0 for v in differences.values())
    exp.atomic(RUN/'training_evidence.json',dict(completed_epochs=4,optimizer_updates=752,
        optimizer_steps=steps,training_cases_per_epoch=24000,validation_cases=6000,
        trainable_parameters=PARAMETERS,parameter_tensors=len(differences),
        changed_parameter_tensors=sum(v>0 for v in differences.values()),
        parameter_only_delta_l2=float(np.sqrt(sum(differences.values()))),
        network_dtype='float32',initialization_fit_samples=24000,initialization_is_training=True,
        initialization_optimizer_updates=0,updated=exp.stamp()))
    del model,system,resume,last,initial
    torch.cuda.empty_cache()

def postprocess():
    state('postprocessing')
    summary=json.loads((RUN/'summary.json').read_text())
    report=dict(updated=exp.stamp(),assistant_review='pending',
        protocol='One acquisition; unchanged four model-only steps; no measured closed loop; center ABS and edge DIFF all <0.07 waves',
        initialization=json.loads((HERE/'preflight.json').read_text()),
        training=json.loads((RUN/'training_evidence.json').read_text()),
        selected_epoch=summary['selected_epoch'],original={},new_noise={})
    report['old_modular_reference']=saved_metrics(SHORT/'exact_local_response/selected_validation.npz')
    report['previous_failed_unified']=saved_metrics(PREVIOUS/'unified_1x/selected_validation.npz')
    table=[]
    for epoch in range(5):
        path=RUN/('baseline_validation.npz' if epoch==0 else f'validation_{epoch:03d}.npz')
        metrics=saved_metrics(path);report['original'][str(epoch)]=metrics
        table.append(dict(epoch=epoch,selected=epoch==summary['selected_epoch'],
            passes=metrics['full_field_passes'],count=6000,rate=metrics['full_field_rate'],
            all7_passes=metrics['all7']['full_field_passes'],all7_count=1333,
            all7_rate=metrics['all7']['full_field_rate'],invalid=metrics['invalid'],
            field_mean=metrics['field_mean']))
    exp.atomic(HERE/'comparison.json',report)
    with (HERE/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=table[0]);writer.writeheader();writer.writerows(table)
    system=RepairSystem();data=system.load_data();indices=np.flatnonzero(~data['train'])
    q=data['q'][indices];seeds=(data['ids'][indices].astype(np.uint64)+1000000007)%(2**32)
    raw=[]
    with torch.no_grad():
        for start in range(0,len(q),128):
            raw.append(system.native_optics.metrics(torch.as_tensor(q[start:start+128],dtype=torch.float64,device='cuda'))['difference'].cpu().numpy())
    raw=np.concatenate(raw)
    for i,seed in enumerate(seeds):raw[i]+=np.random.RandomState(int(seed)).normal(0.,.015,raw[i].shape)
    noise_data=dict(image=system.legacy.nikon35.grid(raw),raw=raw.astype(np.float32),q=q,ids=seeds,
        cohort=data['cohort'][indices],train=np.zeros(6000,dtype=bool),files=data['files'])
    exp.atomic(HERE/'noise_manifest.json',dict(original_validation_indices=indices.tolist(),
        original_seeds=data['ids'][indices].tolist(),new_noise_seeds=seeds.tolist(),noise_std=.015,
        same6000poses=True,independent_scenes=False,selection_on_this_noise=False))
    del data
    for name in ('candidate','last'):
        destination=RUN/('new_noise_'+name+'.npz')
        if not destination.exists():
            if name=='last' and summary['selected_epoch']==4:
                shutil.copyfile(RUN/'new_noise_candidate.npz',destination)
            else:
                state('new_noise_validation',checkpoint=name)
                model=system.model('large')
                saved=torch.load(RUN/(name+'.pt'),map_location='cpu',weights_only=False)
                model.load_state_dict(saved['model'])
                exp.evaluate(system,model,noise_data,np.arange(6000),128,destination)
                del model,saved
        report['new_noise'][name]=saved_metrics(destination)
        exp.atomic(HERE/'comparison.json',report)
    lines=['712万统一网络修复：4轮训练及验证计算完成，等待助手核验解释。',
        '先用24000训练标签拟合完整35维输出头，再进行4轮、752次全网AdamW更新。',
        '初始化拟合属于额外的训练集计算，非额外优化器训练轮次；未使用验证标签初始化。',
        '一次采集，4步模型修正，无实测闭环；中心ABS/边缘DIFF，三场均<0.07 waves。']
    for row in table:
        lines.append(f'Epoch {row["epoch"]}: {row["passes"]}/6000 ({row["rate"]:.2%}); all7 {row["all7_passes"]}/1333 ({row["all7_rate"]:.2%}); mean {row["field_mean"]:.6f} waves')
    for name,m in report['new_noise'].items():
        lines.append(f'New noise {name}: {m["full_field_passes"]}/6000 ({m["full_field_rate"]:.2%}); all7 {m["all7"]["full_field_passes"]}/1333')
    lines.extend(['新噪声测试仍为同一6000个位姿，不代表独立场景。',
        '本次联合修改初始化和直接位姿监督，不能单独归因于其中一项。',
        '程序已生成结果；原模型、失败记录和Trepan保留。'])
    (HERE/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report

def main():
    lock=(HERE/'pipeline.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        assert json.loads((HERE/'preflight.json').read_text())['passed']
        if not (RUN/'training_evidence.json').exists():training()
        report=postprocess()
        state('completed',assistant_review='pending',selected_epoch=report['selected_epoch'])
        with (HERE/'HANDOFF.txt').open('a',encoding='utf-8') as f:
            f.write('\nCOMPLETED '+exp.stamp()+'; program results available, assistant review pending.\n')
    except Exception:
        state('failed',error=traceback.format_exc());raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()

if __name__=='__main__':main()
