"""Full cohort/subset summaries and fixed-noise confirmation, no retraining."""
from unified import *
import csv

def saved_metrics(path):
    with np.load(path) as z:
        f,v,a=z['fields'],z['valid'],z['absolute']
        moving=np.any(z['initial'].reshape(-1,7,5)!=0,axis=2).sum(1)
        result=exp.aggregate(f,v,a,.07)
        result['by_moving_bodies']={str(k):exp.aggregate(f[moving==k],v[moving==k],a[moving==k],.07) for k in np.unique(moving)}
        result['all7']=result['by_moving_bodies']['7']
        assert result['count']==6000 and result['all7']['count']==1333
    return result

def main():
    report=dict(updated=exp.stamp(),assistant_review='pending',plan=json.loads((HERE/'plan.json').read_text()),models={})
    old=SHORT/'exact_local_response'
    report['original_modular_reference']=dict(parameters=REFERENCE_PARAMETERS,
        original=saved_metrics(old/'selected_validation.npz'),
        new_noise=saved_metrics(old/'noise_confirmation/idea_selected.npz'),
        checkpoint=str(old/'candidate.pt'),reused_completed_results=True)
    system=UnifiedSystem('1x');data=system.load_data();indices=np.flatnonzero(~data['train'])
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
    table=[]
    for size in ('1x','2x'):
        folder=HERE/('unified_'+size)
        evidence=json.loads((folder/'training_evidence.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        row=dict(training=evidence,selected_epoch=summary['selected_epoch'],original={})
        for epoch in range(5):
            path=folder/('baseline_validation.npz' if epoch==0 else f'validation_{epoch:03d}.npz')
            metrics=saved_metrics(path);row['original'][str(epoch)]=metrics
            table.append(dict(size=size,parameters=evidence['trainable_parameters'],epoch=epoch,
                selected=epoch==summary['selected_epoch'],full_passes=metrics['full_field_passes'],count=6000,
                full_rate=metrics['full_field_rate'],all7_passes=metrics['all7']['full_field_passes'],
                all7_count=1333,all7_rate=metrics['all7']['full_field_rate'],invalid=metrics['invalid'],field_mean=metrics['field_mean']))
        row['new_noise']={}
        for checkpoint in ('candidate','last'):
            destination=folder/('new_noise_'+checkpoint+'.npz')
            if not destination.exists():
                current=UnifiedSystem(size);model=current.model('large')
                saved=torch.load(folder/(checkpoint+'.pt'),map_location='cpu',weights_only=False)
                model.load_state_dict(saved['model'])
                if checkpoint=='last' and summary['selected_epoch']==4:
                    import shutil
                    shutil.copyfile(folder/'new_noise_candidate.npz',destination)
                else:
                    exp.atomic(HERE/'postprocess_state.json',dict(status='evaluating_new_noise',size=size,checkpoint=checkpoint,updated=exp.stamp()))
                    exp.evaluate(current,model,noise_data,np.arange(6000),128,destination)
                del model,current
            row['new_noise'][checkpoint]=saved_metrics(destination)
        report['models'][size]=row
        exp.atomic(HERE/'comparison.json',report)
    with (HERE/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=table[0]);writer.writeheader();writer.writerows(table)
    oldmetrics=report['original_modular_reference']['original']
    lines=['统一主干大小对照：计算完成，等待助手核验解释',
        '一次观测包、相同4步模型修正、无实测闭环；中心ABS/边缘DIFF，三场均<0.07waves。',
        '原24000训练/6000验证；新噪声仍为同一6000个位姿，不是独立场景。',
        '两个新规模共有初始化来自旧全局分支；不等价于旧完整分组网络的初始函数。',
        '网络容量变化仅通过四个Transformer前馈层宽度调整，其余统一结构相同。',
        f'旧分组参考：混合{oldmetrics["full_field_passes"]}/6000，全7动{oldmetrics["all7"]["full_field_passes"]}/1333。']
    for size,row in report['models'].items():
        lines.append(f'\n{size}：{row["training"]["trainable_parameters"]}参数；4轮、752次更新；选中epoch{row["selected_epoch"]}。')
        for epoch in ('0','4'):
            m=row['original'][epoch]
            lines.append(f'  epoch{epoch}原观测：混合{m["full_field_passes"]}/6000 ({m["full_field_rate"]:.2%})；全7动{m["all7"]["full_field_passes"]}/1333 ({m["all7"]["full_field_rate"]:.2%})。')
        for checkpoint,m in row['new_noise'].items():
            lines.append(f'  {checkpoint}新噪声：混合{m["full_field_passes"]}/6000 ({m["full_field_rate"]:.2%})；全7动{m["all7"]["full_field_passes"]}/1333 ({m["all7"]["full_field_rate"]:.2%})。')
    lines.extend(['','所有原模型与失败记录保留；未自动替换候选，Trepan未动。',
        '本文件由程序生成；后台完成不等于助手已经核验并解释。'])
    (HERE/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    exp.atomic(HERE/'postprocess_state.json',dict(status='completed',assistant_review='pending',updated=exp.stamp()))
    print(json.dumps(dict(status='completed',assistant_review='pending',comparison=str(HERE/'comparison.json'))))

if __name__=='__main__':main()
