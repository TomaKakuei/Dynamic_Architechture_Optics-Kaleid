"""Exactly four epochs after diagnosis, plus full original/noise validation."""
from stable import *
import argparse,ctypes,os,traceback,csv,shutil
RUN=HERE/'stable_training'

def state(status,**extra):
    exp.atomic(HERE/'state.json',dict(status=status,pid=os.getpid(),updated=exp.stamp(),**extra))

def metrics(path):
    with np.load(path) as z:
        f,v,a=z['fields'],z['valid'],z['absolute']
        moving=np.any(z['initial'].reshape(-1,7,5)!=0,axis=2).sum(1)
        result=exp.aggregate(f,v,a,.07)
        result['all7']=exp.aggregate(f[moving==7],v[moving==7],a[moving==7],.07)
        assert result['count']==6000 and result['all7']['count']==1333
        assert np.isfinite(z['command']).all()
        return result

def main():
    lock=(HERE/'pipeline.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        assert json.loads((HERE/'diagnostic_state.json').read_text())['status']=='completed'
        assert json.loads((REPAIR/'state.json').read_text())['status']=='completed'
        system=StableSystem();exp.make_optimizer=optimizer
        args=argparse.Namespace(stage='train',system='nikon',variant='large',epochs=4,batch=128,microbatch=128,seed=SEED)
        state('training',details=str(RUN/'state.json'))
        exp.train(system,args,out=RUN)
        saved=torch.load(RUN/'resume.pt',map_location='cpu',weights_only=False)
        last=torch.load(RUN/'last.pt',map_location='cpu',weights_only=False)
        original=torch.load(system.initialization,map_location='cpu',weights_only=False)['model']
        parameters={k:v for k,v in original.items() if not k.startswith(('base.','native_optics.','feature_center','feature_scale'))}
        changes={k:float((last['model'][k].double()-v.double()).square().sum()) for k,v in parameters.items()}
        steps=sorted({int(v['step']) for v in saved['optimizer']['state'].values()})
        assert len(parameters)==101 and sum(v.numel() for v in parameters.values())==PARAMETERS
        assert steps==[752] and last['epoch']==4 and all(v>0 for v in changes.values())
        evidence=dict(completed_epochs=4,optimizer_updates=752,parameters=PARAMETERS,
            parameter_tensors=101,changed_parameter_tensors=sum(v>0 for v in changes.values()),
            parameter_only_delta_l2=float(np.sqrt(sum(changes.values()))),network_dtype='float32',
            diagnostic_discarded_optimizer_updates=system.decision['diagnostic_discarded_optimizer_updates'],training_cases_per_epoch=24000,validation_count=6000)
        exp.atomic(RUN/'training_evidence.json',evidence)
        summary=json.loads((RUN/'summary.json').read_text())
        report=dict(updated=exp.stamp(),assistant_review='pending',training=evidence,
            decision=system.decision,selected_epoch=summary['selected_epoch'],original={},new_noise={},
            old_modular_reference=metrics(SHORT/'exact_local_response/selected_validation.npz'),
            failed_repair_initial=metrics(REPAIR/'training/baseline_validation.npz'),
            failed_repair_final=metrics(REPAIR/'training/validation_004.npz'))
        table=[]
        for epoch in range(5):
            m=metrics(RUN/('baseline_validation.npz' if epoch==0 else f'validation_{epoch:03d}.npz'))
            report['original'][str(epoch)]=m
            table.append(dict(epoch=epoch,selected=epoch==summary['selected_epoch'],passes=m['full_field_passes'],
                count=6000,rate=m['full_field_rate'],all7_passes=m['all7']['full_field_passes'],all7_count=1333,
                invalid=m['invalid'],field_mean=m['field_mean']))
        exp.atomic(HERE/'comparison.json',report)
        with (HERE/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=table[0]);writer.writeheader();writer.writerows(table)
        del saved,last,original,parameters
        state('noise_data')
        data=system.load_data();ix=np.flatnonzero(~data['train']);q=data['q'][ix]
        seeds=(data['ids'][ix].astype(np.uint64)+1000000007)%(2**32)
        raws=[]
        with torch.no_grad():
            for start in range(0,len(q),128):
                raws.append(system.native_optics.metrics(torch.as_tensor(q[start:start+128],dtype=torch.float64,device='cuda'))['difference'].cpu().numpy())
        raw=np.concatenate(raws)
        for i,seed in enumerate(seeds):raw[i]+=np.random.RandomState(int(seed)).normal(0.,.015,raw[i].shape)
        noise=dict(image=system.legacy.nikon35.grid(raw),raw=raw.astype(np.float32),q=q,ids=seeds,
            cohort=data['cohort'][ix],train=np.zeros(6000,dtype=bool),files=data['files'])
        exp.atomic(HERE/'noise_manifest.json',dict(original_seeds=data['ids'][ix].tolist(),new_seeds=seeds.tolist(),
            same_6000_poses=True,independent_scenes=False,noise_std=.015,used_for_selection=False))
        del data
        for name in ('candidate','last'):
            destination=RUN/('new_noise_'+name+'.npz')
            if not destination.exists():
                if name=='last' and summary['selected_epoch']==4:shutil.copyfile(RUN/'new_noise_candidate.npz',destination)
                else:
                    state('new_noise_validation',checkpoint=name)
                    model=system.model('large')
                    model.load_state_dict(torch.load(RUN/(name+'.pt'),map_location='cpu',weights_only=False)['model'])
                    exp.evaluate(system,model,noise,np.arange(6000),128,destination)
                    del model
            report['new_noise'][name]=metrics(destination)
            exp.atomic(HERE/'comparison.json',report)
        lines=['Nikon统一712万参数：条件修复后的4轮计算完成，等待助手核验。',
            '一次采集，4步原模型修正，无实测闭环；中心ABS/边缘DIFF，全部<0.07waves。',
            '24000训练/6000原验证；同6000位姿新噪声检查不代表独立场景。']
        for r in table:lines.append(f'Epoch {r["epoch"]}: {r["passes"]}/6000 ({r["rate"]:.2%}); all7 {r["all7_passes"]}/1333; mean {r["field_mean"]:.6f} waves')
        for name,m in report['new_noise'].items():lines.append(f'New noise {name}: {m["full_field_passes"]}/6000 ({m["full_field_rate"]:.2%})')
        (HERE/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        state('completed',assistant_review='pending',selected_epoch=summary['selected_epoch'])
    except Exception:
        state('failed',error=traceback.format_exc());raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()

if __name__=='__main__':main()
