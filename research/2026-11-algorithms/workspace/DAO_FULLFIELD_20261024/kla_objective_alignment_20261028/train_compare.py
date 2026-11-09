"""Two sequential four-epoch KLA pilots, same data and optimizer budget.

Fresh candidates only. One measured packet at inference; labels are used only
for supervised optical training/evaluation. No source data generation or hashes.
"""
from pathlib import Path
import sys, os, json, math, traceback
OUT=Path(__file__).resolve().parent/'training_compare'
OUT.mkdir(exist_ok=True)
SOURCE=OUT.parents[2]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0,str(SOURCE))
from common import np, atomic, stamp, clock, awake
from network import restore
from optics import Optics, torch
from geometry import Geometry
sys.path.insert(0,str(OUT.parent))
from diagnose import values, compact
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
ARMS=('old_wfe_stable','psf_stable')
EPOCHS=4; BATCH=32; MICRO=16; LR=3e-5

def state(stage,**kw):
    atomic(OUT/'state.json',dict(stage=stage,pid=os.getpid(),updated=stamp(),**kw))

def save(path,data):
    tmp=path.with_suffix('.tmp');torch.save(data,tmp);os.replace(tmp,path)

def stop():
    return (OUT/'STOP_REQUESTED.json').exists()

def summary(a):
    valid=a['valid']; good=valid&(a['score']>=.45)
    return dict(count=len(valid),passes=int(good.sum()),rate=float(good.mean()),invalid=int((~valid).sum()),
        q_mean=float(a['score'].mean()),wrms_mean=float(a['wrms'].mean()),
        psf_hellinger_mean=float(a['psf_hellinger'].mean()))

def ranking(a):
    return (-a['passes'],-a['q_mean'])

def main():
    if (OUT/'complete.json').exists():
        print((OUT/'complete.json').read_text());return
    if stop():state('stopped_before_start');return
    awake();tick=clock()
    with np.load(SOURCE/'cache/metadata.npz') as z:meta={k:z[k].copy() for k in z.files}
    with np.load(SOURCE/'training/joint_validation3000.npz') as z:old={k:z[k].copy() for k in z.files}
    images=np.load(SOURCE/'cache/images.npy',mmap_mode='r')
    plan_path=OUT/'selection.npz'
    if not plan_path.exists():
        rng=np.random.default_rng(2026102819)
        train=np.concatenate([rng.choice(np.where((meta['cohort']==c)&(meta['indices']<9000))[0],1000,replace=False) for c in range(3)])
        diagnostic=set(json.loads((OUT.parent/'protocol.json').read_text())['rows'])
        valrows=np.concatenate([rng.choice([i for i in np.where(old['cohort']==c)[0] if i not in diagnostic],200,replace=False) for c in range(3)])
        np.savez(plan_path,train=train,valrows=valrows)
    with np.load(plan_path) as z:train=z['train'];valrows=z['valrows']
    protocol=dict(arms=list(ARMS),epochs_per_arm=EPOCHS,training_samples=3000,training_per_cohort=1000,
        selection_samples=600,selection_per_cohort=200,selection_scope='Reused original selection validation; exclude48 diagnostic rows, not an independent test',
        initial_checkpoint=str(SOURCE/'training/candidate.pt'),initial_epoch=19,
        objective_old='Original mean ell/(1+ell), ell=log1p((PT WRMS/0.07)^2)',
        objective_psf='Unit-flux nominal PSF squared Hellinger distance, mean over all9 fields; same fixed image plane, defocus retained, PT removed',
        common_auxiliary='0.05 SmoothL1 inverse pose +10 original mechanical penalty; original invalid/path treatment',
        lr=LR,lr_schedule='constant for4epochs',optimizer='fresh AdamW, weight_decay1e-4, clip_grad_norm5',
        batch=BATCH,microbatch=MICRO,training_grid=65,evaluation_grid=65,
        shared_stability_changes='Lower LR; freeze BatchNorm running statistics at epoch19, keep affine weights trainable; grid65 shared by both arms',
        frozen_bn_uses_batch_statistics_at_inference=False,network_dtype='float32',optics_dtype='float64',
        same_orders_and_seed=True,automatic_full3000='Only best candidate if at least+12/600 passes, no additional invalid, and lower PSF error',
        automatic_further_training=False,automatic_deployment=False,truth_in_inference=False,new_acquisitions=0,
        original_sources_weights_data_read_only=True,updated=stamp())
    atomic(OUT/'protocol.json',protocol)
    geometry=Geometry();optic=Optics(grid=65,device='cuda')
    with torch.no_grad():ref=values(optic,torch.zeros((1,40),device='cuda',dtype=torch.float64))
    @torch.no_grad()
    def evaluate(model,rows,filename,grid=65,reuse_commands=None):
        op=optic if grid==65 else Optics(grid=grid,device='cuda')
        nominal=ref if grid==65 else values(op,torch.zeros((1,40),device='cuda',dtype=torch.float64))
        pieces=[];commands=[]
        if model is not None:model.eval()
        size=16 if grid==65 else 4
        for first in range(0,len(rows),size):
            ix=rows[first:first+size];ids=old['indices'][ix]
            cmd=torch.tensor(reuse_commands[first:first+size],device='cuda') if reuse_commands is not None else model.command(torch.tensor(images[ids].astype(np.float32),device='cuda'))
            ep=torch.tensor(old['initial'][ix],device='cuda',dtype=torch.float64)+cmd.double()
            v=values(op,ep,nominal);a=compact(v)
            a['valid'] &= np.array([geometry.path(b,a0) for a0,b in zip(old['initial'][ix],ep.cpu().numpy())])
            pieces.append(a);commands.append(cmd.cpu().numpy())
        a={k:np.concatenate([x[k] for x in pieces]) for k in pieces[0]}
        a['command']=np.concatenate(commands);a['rows']=rows;a['indices']=old['indices'][rows]
        np.savez_compressed(filename,**a)
        return summary(a),a
    baseline_path=OUT/'baseline600.npz'
    if baseline_path.exists():
        with np.load(baseline_path) as z:baseline_arrays={k:z[k].copy() for k in z.files}
        baseline=summary(baseline_arrays)
    else:
        state('baseline_psf600')
        baseline,baseline_arrays=evaluate(None,valrows,baseline_path,reuse_commands=old['command'][valrows])
    atomic(OUT/'baseline600.json',baseline)
    histories={};elapsed_arms={}
    for arm in ARMS:
        dest=OUT/arm;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            done=json.loads((dest/'complete.json').read_text());histories[arm]=done;continue
        torch.manual_seed(2026102819);torch.cuda.manual_seed_all(2026102819)
        model,payload=restore(SOURCE/'training/candidate.pt')
        optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
        first_epoch=1;first_batch=0;history=[];best=None;accum=np.zeros(3);processed=0;active=0.
        if (dest/'resume.pt').exists():
            r=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
            model.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer'])
            first_epoch=r['next_epoch'];first_batch=r['next_batch'];history=r['history'];best=r['best'];accum=np.array(r['accum']);processed=r['processed'];active=r['active_seconds']
        def package():
            return dict(model=model.state_dict(),reference=payload['reference'],epoch=epoch,arm=arm,
                source_epoch=19,protocol=protocol,scales=payload.get('scales'),validation=history[-1] if history else None)
        def checkpoint(ep,ba):
            save(dest/'resume.pt',package()|dict(optimizer=optimizer.state_dict(),next_epoch=ep,next_batch=ba,
                history=history,best=best,accum=accum.tolist(),processed=processed,active_seconds=active))
        batches=math.ceil(len(train)/BATCH)
        for epoch in range(first_epoch,EPOCHS+1):
            order=np.random.default_rng(2026102800+epoch).permutation(train)
            model.train()
            for m in model.modules():
                if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):m.eval()
            for batch in range(first_batch,batches):
                if stop():checkpoint(epoch,batch);state('paused_by_stop',arm=arm,epoch=epoch,batch=batch);return
                bt=clock();ids=order[batch*BATCH:(batch+1)*BATCH];optimizer.zero_grad(set_to_none=True);parts=np.zeros(3)
                for micro in range(0,len(ids),MICRO):
                    ix=ids[micro:micro+MICRO];target=torch.tensor(meta['q'][ix],device='cuda',dtype=torch.float64)
                    cmd=model.command(torch.tensor(images[ix].astype(np.float32),device='cuda'));endpoint=target+cmd.double()
                    v=values(optic,endpoint,ref);physical=v['bounded_wfe' if arm=='old_wfe_stable' else 'psf_hellinger']
                    path=torch.tensor([geometry.path(b,a) for a,b in zip(target.detach().cpu().numpy(),endpoint.detach().cpu().numpy())],device='cuda')
                    valid=v['valid']&path&torch.isfinite(physical)
                    physical=torch.where(valid,physical,2+torch.nan_to_num(physical,nan=1.,posinf=1.))
                    pose=torch.nn.functional.smooth_l1_loss(cmd,-target.float(),reduction='none').mean(1)
                    mechanical=geometry.penalty(endpoint);loss=(physical+.05*pose+10*mechanical).mean()
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stop before optimizer update')
                    (loss*(len(ix)/len(ids))).backward()
                    parts+=len(ix)*np.array([float(loss.detach()),float(physical.detach().mean()),float(v['wrms'].detach().mean())])
                    del v,loss,physical,cmd,endpoint
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
                torch.cuda.synchronize();active+=clock()-bt;accum+=parts;processed+=len(ids)
                if (batch+1)%20==0 or batch+1==batches:
                    checkpoint(epoch,batch+1)
                if (batch+1)%10==0 or batch==0 or batch+1==batches:
                    completed_updates=(epoch-1)*batches+batch+1
                    remaining=(EPOCHS*batches-completed_updates)+(len(ARMS)-ARMS.index(arm)-1)*EPOCHS*batches
                    rate=active/max(completed_updates,1)
                    state('training',arm=arm,epoch=epoch,total_epochs=EPOCHS,batch=batch+1,batches=batches,
                        active_seconds=active,seconds_per_update=rate,estimated_remaining_train_seconds=remaining*rate,
                        loss=(accum/processed).tolist(),network_dtype='float32',peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
                    print(json.dumps(dict(arm=arm,epoch=epoch,batch=batch+1,batches=batches,seconds_per_update=rate)),flush=True)
            state('evaluating600',arm=arm,epoch=epoch)
            metric,arrays=evaluate(model,valrows,dest/f'epoch{epoch}_validation600.npz')
            history.append(dict(epoch=epoch,train_loss=(accum/processed).tolist(),validation=metric,train_active_seconds=active))
            atomic(dest/'history.json',history)
            if best is None or ranking(metric)<tuple(best):
                best=ranking(metric);save(dest/'candidate.pt',package()|dict(selected_epoch=epoch,selected_validation=metric))
            save(dest/'last.pt',package())
            first_batch=0;accum=np.zeros(3);processed=0;checkpoint(epoch+1,0)
            print(json.dumps(dict(arm=arm,epoch=epoch,validation=metric)),flush=True)
        selected=torch.load(dest/'candidate.pt',map_location='cpu',weights_only=False)
        done=dict(status='complete',epochs=EPOCHS,selected_epoch=selected['selected_epoch'],validation=selected['selected_validation'],history=history,training_active_seconds=active)
        atomic(dest/'complete.json',done);histories[arm]=done
        del model,optimizer,selected;torch.cuda.empty_cache()
    winner=min(ARMS,key=lambda a:ranking(histories[a]['validation']))
    v=histories[winner]['validation']
    qualifies=v['passes']>=baseline['passes']+12 and v['invalid']<=baseline['invalid'] and v['psf_hellinger_mean']<baseline['psf_hellinger_mean']
    result=dict(status='complete',baseline=baseline,arms=histories,winner_on_selection600=winner,
        full3000_gate_passed=qualifies,full3000=None,automatic_deployment=False,assistant_interpretation='pending')
    atomic(OUT/'pilot_result.json',result)
    if qualifies:
        state('evaluating_selected3000_grid129',arm=winner,total=3000)
        model,_=restore(OUT/winner/'candidate.pt')
        metric,a=evaluate(model,np.arange(len(old['indices'])),OUT/'selected_validation3000_grid129.npz',grid=129)
        # Original refinement includes both the near-threshold set and every
        # grid65/grid129 threshold disagreement.
        with torch.no_grad():
            coarse=[]
            for first in range(0,len(a['command']),16):
                ep=old['initial'][first:first+16]+a['command'][first:first+16]
                m=optic.metrics(torch.tensor(ep,device='cuda',dtype=torch.float64));coarse.append(m['score'].cpu().numpy())
            coarse=np.concatenate(coarse)
            dense=Optics(grid=257,device='cuda');border=np.where((abs(a['score']-.45)<.02)|((a['score']>=.45)!=(coarse>=.45)))[0];refined=a['score'].copy();valid=a['valid'].copy()
            for first in range(0,len(border),2):
                ix=border[first:first+2];ep=old['initial'][ix]+a['command'][ix]
                m=dense.metrics(torch.tensor(ep,device='cuda',dtype=torch.float64))
                refined[ix]=m['score'].cpu().numpy();valid[ix]&=m['valid'].cpu().numpy()
            np.savez_compressed(OUT/'selected_refined_scores.npz',score=refined,valid=valid,border=border,coarse65=coarse)
        result['full3000']=dict(grid129=metric,refined_passes=int((valid&(refined>=.45)).sum()),
            refined_rate=float((valid&(refined>=.45)).mean()),invalid=int((~valid).sum()),refined_cases=len(border),
            original_selected_epoch19_refined_passes=775,scope='Same original3000 selection validation, not independent test')
    result['active_elapsed_seconds']=clock()-tick
    atomic(OUT/'complete.json',result)
    state('complete',winner=winner,full3000_gate_passed=qualifies,active_seconds=clock()-tick,assistant_interpretation='pending')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:awake(False)
