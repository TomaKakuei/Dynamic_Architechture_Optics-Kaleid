"""Four full epochs of the original enlarged Nikon route with fixed BN statistics."""
from pathlib import Path
import sys,os,json,math,ctypes,traceback,shutil
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'original_routes_40ep_20261025'))
import worker as old
from worker import torch,np,exp,atomic
from exact_refinement import ExactResponseNetwork
TRIAL='fixed_bn_4ep'


class FixedStatsNetwork(ExactResponseNetwork):
    def train(self,mode=True):
        super().train(mode)
        # Keep the pretrained sensor normalization fixed; affine parameters
        # and every other neural parameter remain trainable.
        for module in self.modules():
            if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        return self


class ExactTrajectoryNetwork(FixedStatsNetwork):
    def forward(self,image,raw=None):
        if not torch.is_grad_enabled():return super().forward(image,raw)
        from trajectory_backward import four_steps
        prediction=exp.LargerNetwork.forward(self,image,raw)
        prior=(prediction*self.base.target_std+self.base.target_mean).clamp(-1.5,1.5).double()
        physical=four_steps(prior,raw.flatten(1).double()/7.,self)
        return ((physical-self.base.target_mean)/self.base.target_std).float()


def train():
    out=HERE/TRIAL;out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return
    def state(status,**kw):
        atomic(out/'state.json',dict(status=status,pid=os.getpid(),updated=exp.stamp(),**kw))
    if (HERE/'STOP').exists():
        state('paused',reason='Local STOP marker');return
    state('loading')
    system=old.make_system('nikon');data=old.load_data(system)
    train_ids=np.flatnonzero(data['train']);val_ids=np.flatnonzero(~data['train'])
    seed=2026102501;batch=128;epochs=4
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,_=old.make_model(system)
    source=old.HERE/'nikon/candidate.pt'
    initial=torch.load(source,map_location='cpu',weights_only=False)
    assert initial['epoch']==0 and initial['validation']['full_field_passes']==1726
    model.load_state_dict(initial['model'])
    # This changes train-mode normalization only; inference stays identical.
    model.eval()
    ix=train_ids[:4]
    with torch.no_grad():before=system.forward(model,data,ix)[2].clone()
    model.__class__=FixedStatsNetwork if TRIAL=='fixed_bn_4ep' else ExactTrajectoryNetwork
    model.train()
    with torch.no_grad():after=system.forward(model,data,ix)[2]
    torch.testing.assert_close(after,before,rtol=1e-4,atol=1e-5)
    bn_buffers={k:v.clone() for k,v in model.named_buffers() if k.endswith(('running_mean','running_var','num_batches_tracked'))}
    assert len(bn_buffers)==63
    assert all(p.requires_grad for p in model.parameters())
    optimizer=exp.make_optimizer(model)
    config=dict(system='nikon',trial=TRIAL,epochs=epochs,training_count=len(train_ids),
        validation_count=len(val_ids),batch=batch,seed=seed,parameters=sum(p.numel() for p in model.parameters()),
        parameter_dtype='float32',source_checkpoint=str(source),source_epoch=0,
        only_numerical_change='Freeze 21 BN running-statistic layers in both training and inference; all affine/other parameters trainable',
        original_loss=True,original_learning_rates=[1e-5,1e-4],original_cosine_horizon=40,
        inference_unchanged=True,one_acquisition=True,model_refinements=4,measured_closed_loop=False,
        metric='Center ABS + both edges DIFF; all three <0.07 waves',
        original_validation_reused_as_epoch0=True,train_mode_command_max_diff=float((after-before).abs().max()))
    atomic(out/'config.json',config)
    if TRIAL=='trajectory_4ep':
        check=json.loads((HERE/'trajectory_branch_acceptance.json').read_text())
        assert check['status']=='passed', 'Real Nikon forward/derivative check must pass before training'
        assert check['tolerance_unchanged'] and check['all_recorded_solver_and_worst_field_branches_unchanged']
        config['only_numerical_change']='Fixed BN statistics + reverse derivative through the actual four accepted GN steps, including optical Hessian-vector products and caps; forward unchanged'
        config['backward']='Exact derivative of chosen smooth four-step branch, verified on real training samples'
        config['derivative_check']='trajectory_branch_acceptance.json; initial larger-step check failure preserved separately'
        atomic(out/'config.json',config)
    atomic(out/'data_manifest.json',dict(training_count=28000,original_training_count=24000,
        added_training_count=4000,all_added_in_training=True,validation_count=6000,original_validation_unchanged=True,
        new_data=str(old.HERE/'nikon_data_v2'),files=data['files']))
    epoch,next_batch,accumulated,processed,elapsed=1,0,0.,0,0.
    history=[];best=None
    if (out/'resume.pt').exists():
        resume=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        assert resume['config']==config
        model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        epoch,next_batch,accumulated,processed,elapsed=[resume[k] for k in ('epoch','next_batch','accumulated','processed','elapsed')]
        history,best=resume['history'],resume['best']
        torch.set_rng_state(resume['rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        del resume
    def payload():return dict(model=model.state_dict(),config=config,system='nikon',variant=TRIAL)
    def checkpoint():
        exp.save_pt(out/'resume.pt',payload()|dict(optimizer=optimizer.state_dict(),epoch=epoch,next_batch=next_batch,
            accumulated=accumulated,processed=processed,elapsed=elapsed,history=history,best=best,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    def paused():
        if not (HERE/'STOP').exists():return False
        checkpoint();state('paused',epoch=epoch,batch=next_batch,epochs=epochs)
        return True
    if not history:
        baseline=initial['validation'];best=exp.selection(baseline)
        history=[dict(epoch=0,validation=baseline,active_seconds=0.,reused_verified_identical_baseline=True)]
        shutil.copyfile(old.HERE/'nikon/baseline_validation.npz',out/'baseline_validation.npz')
        exp.save_pt(out/'candidate.pt',payload()|dict(epoch=0,validation=baseline))
        atomic(out/'selected.json',dict(epoch=0,validation=baseline));exp.curves(out,history);checkpoint()
    del initial
    batches=math.ceil(len(train_ids)/batch)
    while epoch<=epochs:
        order=np.random.default_rng(seed+epoch).permutation(train_ids)
        factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/40))
        for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
        for b in range(next_batch,batches):
            if paused():return
            tick=exp.active_time();model.train();optimizer.zero_grad(set_to_none=True)
            ix=order[b*batch:(b+1)*batch]
            loss,parts=system.loss(model,data,ix)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss: '+repr(parts))
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            elapsed+=exp.active_time()-tick;accumulated+=float(loss.detach())*len(ix);processed+=len(ix);next_batch=b+1
            if next_batch%25==0 or next_batch==batches:checkpoint()
            if next_batch%10==0 or next_batch==1 or next_batch==batches:
                state('training',epoch=epoch,epochs=epochs,batch=next_batch,batches=batches,
                    loss=accumulated/processed,gradient_norm=float(norm),epoch_active_seconds=elapsed,
                    remaining_training_seconds=elapsed/next_batch*(batches-next_batch+(epochs-epoch)*batches),
                    peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
                print(f'{TRIAL} epoch{epoch}/4 batch{next_batch}/{batches} loss{accumulated/processed:.7f}',flush=True)
        if paused():return
        state('validation',epoch=epoch,epochs=epochs)
        tick=exp.active_time()
        validation=exp.evaluate(system,model,data,val_ids,batch,out/f'validation_{epoch:03d}.npz')
        elapsed+=exp.active_time()-tick
        current_buffers=dict(model.named_buffers())
        assert all(torch.equal(current_buffers[k],v) for k,v in bn_buffers.items()),'Frozen BN state changed'
        if exp.selection(validation)<tuple(best):
            best=exp.selection(validation)
            exp.save_pt(out/'candidate.pt',payload()|dict(epoch=epoch,validation=validation))
            atomic(out/'selected.json',dict(epoch=epoch,validation=validation))
        history.append(dict(epoch=epoch,loss=accumulated/processed,validation=validation,active_seconds=elapsed,
            bn_buffers_unchanged=True,all_parameters_trainable=True))
        exp.curves(out,history);exp.save_pt(out/'last.pt',payload()|dict(epoch=epoch,validation=validation))
        print(json.dumps(history[-1]),flush=True)
        epoch+=1;next_batch=0;accumulated=0.;processed=0;elapsed=0.;checkpoint()
    selected=json.loads((out/'selected.json').read_text())
    source_file=out/('baseline_validation.npz' if selected['epoch']==0 else f"validation_{selected['epoch']:03d}.npz")
    shutil.copyfile(source_file,out/'selected_validation.npz')
    atomic(out/'summary.json',dict(status='completed',assistant_review='pending',epochs=4,optimizer_updates=4*batches,
        parameters=config['parameters'],baseline=history[0]['validation'],selected_epoch=selected['epoch'],
        selected_validation=selected['validation'],last_validation=history[-1]['validation'],
        active_seconds=sum(r['active_seconds'] for r in history),bn_buffers_unchanged=True,independent_test=False,
        comparison_to_old_first4=json.loads((old.HERE/'nikon/history.json').read_text())[:5],updated=exp.stamp()))
    state('completed',epoch=4,epochs=4,selected_epoch=selected['epoch'],assistant_review='pending')
    from review_predictions import main as review_saved_predictions
    review_saved_predictions()


def main():
    global TRIAL
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--trial',choices=['fixed_bn_4ep','trajectory_4ep'],default='fixed_bn_4ep')
    TRIAL=parser.parse_args().trial
    import msvcrt
    lock=(HERE/'training.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    started=dict(pid=os.getpid(),started=exp.stamp(),trial=TRIAL)
    atomic(HERE/'started.json',started)
    atomic(HERE/f'{TRIAL}_started.json',started)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:train()
    except Exception:
        atomic(HERE/'training_failure.json',dict(error=traceback.format_exc(),updated=exp.stamp()));raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000);lock.close()


if __name__=='__main__':main()
