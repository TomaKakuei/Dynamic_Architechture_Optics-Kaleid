"""Four additional Nikon epochs; only add direct pre-solver pose supervision.

Continue the completed additive E4 weights, AdamW state and LR schedule.
Reuse every observation. Final evaluation is the original6000, once, with
resumable chunks. No preliminary/per-epoch evaluation or independent replay.
"""
from pathlib import Path
import os,sys,json,math,traceback,ctypes
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SOURCE=ROOT/'nikon_additive_4ep_20261025'
sys.path.insert(0,str(SOURCE))
import run_trial as previous
np,torch,exp=previous.np,previous.torch,previous.exp
nn=torch.nn
EPOCHS,BATCH,PRIOR_WEIGHT=4,128,.02
SEED=previous.SEED


def state(status,**kw):
    exp.atomic(HERE/'state.json',dict(status=status,pid=os.getpid(),updated=exp.stamp(),**kw))


class PriorSupervisedNetwork(previous.AdditiveNetwork):
    def forward(self,image,raw=None,extra=None):
        if raw is None or raw.shape[1:]!=(3,49) or extra is None or extra.shape!=(len(raw),393):
            raise ValueError('Expected original147 DIFF plus393 added observations')
        prediction=exp.LargerNetwork.forward(self,image,raw)
        correction=self.extra_branch((extra-self.extra_mean)/self.extra_std)
        normalized_prior=prediction+correction
        # Training-only graph reference; neither targets nor scores enter forward.
        self.prior_for_loss=normalized_prior if self.training and torch.is_grad_enabled() else None
        prior=(normalized_prior*self.base.target_std+self.base.target_mean).clamp(-1.5,1.5).double()
        physical=previous.four_steps(prior,raw.flatten(1).double()/7.,self)
        return ((physical-self.base.target_mean)/self.base.target_std).float()


class PriorSupervisedSystem(previous.AdditiveSystem):
    def loss(self,model,data,indices):
        original,parts=super().loss(model,data,indices)
        if model.prior_for_loss is None:raise RuntimeError('Missing training prior')
        target=torch.as_tensor(data['q'][indices],device='cuda',dtype=torch.float64)
        normalized=((target-model.base.target_mean)/model.base.target_std).float()
        auxiliary=nn.functional.smooth_l1_loss(model.prior_for_loss,normalized,beta=.25)
        model.prior_for_loss=None
        parts.update(original_total=float(original.detach()),prior_pose=float(auxiliary.detach()),
                     weighted_prior_pose=float((PRIOR_WEIGHT*auxiliary).detach()))
        return original+PRIOR_WEIGHT*auxiliary,parts


def make_model(system,saved):
    stats=dict(mean=saved['extra_mean'].numpy(),std=saved['extra_std'].numpy())
    model=previous.make_model(system,stats)
    model.load_state_dict(saved,strict=True)
    model.__class__=PriorSupervisedNetwork
    model.prior_for_loss=None
    return model


@torch.no_grad()
def evaluate6000(system,model,data):
    out=HERE/'evaluation6000';out.mkdir(exist_ok=True)
    if (out/'result.json').exists():return json.loads((out/'result.json').read_text())
    ix=np.flatnonzero(~data['train'])
    if len(ix)!=6000:raise RuntimeError('Expected original6000 validation')
    exp.atomic(out/'protocol.json',dict(count=6000,original_ids=data['ids'][ix].tolist(),
        new_training_epochs=4,source_training_epochs=4,criterion='All3<0.07waves and valid',
        channels=['center_ABS','minus0.1mm_DIFF','plus0.1mm_DIFF'],audit_nodes=197,
        checkpoint='last.pt',measurements=1,model_only_steps=4,measured_feedback=False,
        independent_review=False,baseline_predictions='Reuse additiveE4 saved original6000'))
    model.eval();blocks=[];tick=exp.active_time();new_count=0
    for start in range(0,len(ix),BATCH):
        if (HERE/'STOP').exists():
            state('paused',stage='evaluation6000',completed=start,total=6000);return None
        rows=ix[start:start+BATCH];path=out/f'part_{start//BATCH:03d}.npz'
        if path.exists():
            with np.load(path) as z:
                if not np.array_equal(z['original_ids'],data['ids'][rows]):raise RuntimeError('Cached evaluation IDs differ')
                block={k:z[k].copy() for k in ('fields','valid','absolute','command')}
        else:
            endpoint,initial,command,_=system.forward(model,data,rows)
            fields,valid,absolute=system.metrics(endpoint,initial)
            block=dict(fields=fields.cpu().numpy(),valid=valid.cpu().numpy(),
                       absolute=absolute.cpu().numpy(),command=command.cpu().numpy())
            np.savez(path,original_ids=data['ids'][rows],**block);new_count+=len(rows)
        blocks.append(block)
        elapsed=exp.active_time()-tick;completed=start+len(rows)
        state('evaluating6000',completed=completed,total=6000,new_predictions_this_invocation=new_count,
              active_evaluation_seconds=elapsed,remaining_evaluation_seconds=elapsed/max(1,new_count)*(6000-completed))
    values={key:np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    f,v,a=values['fields'],values['valid'],values['absolute']
    summary=exp.aggregate(f,v,a,.07)
    summary['cohorts']={str(c):exp.aggregate(f[data['cohort'][ix]==c],v[data['cohort'][ix]==c],
        a[data['cohort'][ix]==c],.07) for c in np.unique(data['cohort'][ix])}
    active=(np.max(np.abs(data['q'][ix].reshape(-1,7,5)),axis=2)>1e-10).sum(1)
    summary['active_bodies']={str(c):exp.aggregate(f[active==c],v[active==c],a[active==c],.07) for c in np.unique(active)}
    np.savez(out/'final_validation6000.npz',indices=ix,original_ids=data['ids'][ix],
             cohort=data['cohort'][ix],initial=data['q'][ix],**values)
    baseline_path=SOURCE/'evaluation6000/final_validation6000.npz'
    with np.load(baseline_path) as z:
        lookup={int(identity):i for i,identity in enumerate(z['original_ids'])}
        order=np.asarray([lookup[int(identity)] for identity in data['ids'][ix]])
        old_f=z['fields'][order];old_v=z['valid'][order];old_a=z['absolute'][order]
    old_pass=old_v&np.isfinite(old_f).all(1)&(old_f.max(1)<.07)
    new_pass=v&np.isfinite(f).all(1)&(f.max(1)<.07)
    report=dict(status='completed',validation=summary,baseline_additive_E4=exp.aggregate(old_f,old_v,old_a,.07),
        baseline_source=str(baseline_path),baseline_new_inference=False,
        change=dict(passes=int(new_pass.sum()-old_pass.sum()),
            percentage_points=float(100*(new_pass.mean()-old_pass.mean())),
            gained=int((new_pass&~old_pass).sum()),lost=int((~new_pass&old_pass).sum())),
        new_training_epochs=4,independent_review=False,automatic_deployment=False,
        active_seconds_this_invocation=exp.active_time()-tick,updated=exp.stamp())
    exp.atomic(out/'result.json',report)
    return report


def main():
    import msvcrt
    lock=(HERE/'training.lock').open('a+b');lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (HERE/'summary.json').exists():return
    if (HERE/'STOP').exists():state('paused',reason='Local STOP');return
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        state('loading')
        system=previous.old.make_system('nikon');system.__class__=PriorSupervisedSystem
        data=previous.old.load_data(system)
        data['extra']=np.load(SOURCE/'data/extra.npy',mmap_mode='r')
        train_ids=np.flatnonzero(data['train']);batches=math.ceil(len(train_ids)/BATCH)
        if len(train_ids)!=28000 or len(data['q'])!=34000:raise RuntimeError('Expected28000train+6000validation')
        own_resume=(HERE/'resume.pt').exists()
        source=HERE/'resume.pt' if own_resume else SOURCE/'resume.pt'
        saved=torch.load(source,map_location='cpu',weights_only=False)
        if not own_resume and (saved['epoch']!=5 or saved['next_batch']!=0):
            raise RuntimeError('Source must be completed additiveE4')
        model=make_model(system,saved['model'])
        optimizer=exp.make_optimizer(model);optimizer.load_state_dict(saved['optimizer'])
        config=dict(trial='direct_prior_supervision',system='nikon',new_epochs=4,source_completed_epochs=4,
            source_checkpoint=str(SOURCE/'resume.pt'),source_optimizer='Continue completed E4 AdamW moments',
            neural_parameters=sum(p.numel() for p in model.parameters()),neural_dtype='float32',
            all_neural_parameters_trainable=True,batch=BATCH,seed=SEED,
            training_cases=28000,original_training_cases=24000,added_cases=4000,all_added_in_training=True,
            validation_cases=6000,evaluation_epochs=[4],baseline='Saved additive E4 original6000:1825passes',
            prior_loss=dict(type='SmoothL1',beta=.25,weight=PRIOR_WEIGHT,location='Normalized output before solver/clamp',
                labels='Training pose only; never passed to model.forward'),
            existing_loss='Original endpoint physical +0.02 endpoint pose; unchanged',
            inverse='Original147 DIFF;4steps;ridge1e-4;cap0.25;gains1,.5,.25;corrected trajectory VJP',
            inputs='Original147 DIFF + cached center197ABS and outer4x49DIFF; same normalization',
            extra_branch='Existing trained175651parameter branch retained',
            measurements=1,measured_feedback=False,bn_statistics='Frozen',
            optimizer='Existing AdamW, weight_decay1e-4, clip5',initial_lrs=[1e-5,1e-4],
            lr_schedule='Continue cosine horizon40 at global epochs5..8, local epochs1..4',
            independent_review=False,preliminary_evaluation=False,per_epoch_evaluation=False,
            new_observations_generated=0,automatic_followup_training=False,automatic_deployment=False)
        if own_resume and saved['config']!=config:raise RuntimeError('Resume config differs')
        exp.atomic(HERE/'config.json',config)
        epoch,next_batch,total_seconds=1,0,0.
        loss_sum,processed,epoch_seconds=0.,0,0.
        part_sums={};history=[]
        if own_resume:
            epoch,next_batch,total_seconds,loss_sum,processed,epoch_seconds,part_sums,history=[saved[k] for k in
                ('epoch','next_batch','total_seconds','loss_sum','processed','epoch_seconds','part_sums','history')]
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);del saved
        baseline=json.loads((SOURCE/'evaluation6000/result.json').read_text())['validation']
        exp.atomic(HERE/'baseline6000.json',dict(source=str(SOURCE/'evaluation6000/result.json'),
                                              reused=True,new_predictions=0,validation=baseline))
        def checkpoint():
            exp.save_pt(HERE/'resume.pt',dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch,next_batch=next_batch,total_seconds=total_seconds,loss_sum=loss_sum,
                processed=processed,epoch_seconds=epoch_seconds,part_sums=part_sums,history=history,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        if not own_resume:checkpoint()
        while epoch<=EPOCHS:
            global_epoch=4+epoch
            order=np.random.default_rng(SEED+global_epoch).permutation(train_ids)
            factor=.1+.9*.5*(1+math.cos(math.pi*(global_epoch-1)/40))
            for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
            for b in range(next_batch,batches):
                if (HERE/'STOP').exists():checkpoint();state('paused',epoch=epoch,batch=next_batch,epochs=4);return
                tick=exp.active_time();model.train();optimizer.zero_grad(set_to_none=True)
                indices=order[b*BATCH:(b+1)*BATCH]
                loss,parts=system.loss(model,data,indices)
                if not bool(torch.isfinite(loss)):raise RuntimeError(f'Nonfinite loss:{parts}')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                elapsed=exp.active_time()-tick;total_seconds+=elapsed;epoch_seconds+=elapsed
                loss_sum+=float(loss.detach())*len(indices);processed+=len(indices);next_batch=b+1
                for key,value in parts.items():part_sums[key]=part_sums.get(key,0.)+float(value)*len(indices)
                if next_batch%25==0 or next_batch==batches:checkpoint()
                if next_batch%5==0 or next_batch==1 or next_batch==batches:
                    updates=(epoch-1)*batches+next_batch
                    status=dict(epoch=epoch,epochs=4,batch=next_batch,batches=batches,
                        optimizer_updates=updates,total_optimizer_updates=4*batches,loss=loss_sum/processed,
                        mean_parts={k:v/processed for k,v in part_sums.items()},last_parts=parts,
                        gradient_norm=float(grad),learning_rates=[g['lr'] for g in optimizer.param_groups],
                        active_training_seconds=total_seconds,seconds_per_batch=total_seconds/updates,
                        remaining_training_seconds=total_seconds/updates*(4*batches-updates),
                        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        baseline6000_passes=baseline['full_field_passes'])
                    state('training',**status);print(json.dumps(status),flush=True)
            history.append(dict(epoch=epoch,loss=loss_sum/processed,active_seconds=epoch_seconds,
                                parts={k:v/processed for k,v in part_sums.items()}))
            exp.atomic(HERE/'history.json',history)
            epoch+=1;next_batch=0;loss_sum=0.;processed=0;epoch_seconds=0.;part_sums={};checkpoint()
        if not (HERE/'last.pt').exists():exp.save_pt(HERE/'last.pt',dict(config=config,model=model.state_dict(),epoch=4))
        if (HERE/'STOP').exists():state('paused',stage='before_evaluation6000',epoch=4);return
        state('evaluating6000',completed=0,total=6000)
        evaluation=evaluate6000(system,model,data)
        if evaluation is None:return
        result=dict(status='completed',epochs=4,optimizer_updates=4*batches,
                    active_training_seconds=total_seconds,evaluation=evaluation,
                    independent_review=False,automatic_deployment=False,assistant_interpretation='pending',updated=exp.stamp())
        exp.atomic(HERE/'summary.json',result)
        state('completed',epoch=4,epochs=4,optimizer_updates=4*batches,
              baseline_passes=baseline['full_field_passes'],final_passes=evaluation['validation']['full_field_passes'])
        print(json.dumps(result),flush=True)
    except BaseException:
        state('failed',error=traceback.format_exc());raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':main()
