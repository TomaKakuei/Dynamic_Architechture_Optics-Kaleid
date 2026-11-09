"""Four epochs: preserve the original Nikon predictor, append measured features.

The old CNNs, normalization, raw attention, head, 147-value inverse residual,
and three-field objective are retained. Only a zero-output neural correction
branch is added. No independent replay, derivative audit or full validation.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import sys
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (ROOT, ROOT/'nikon_short_training', ROOT/'original_routes_40ep_20261025',
                  ROOT/'nikon_large_debug_20261025'):
    sys.path.insert(0, str(directory))
import ctypes
import gc
import json
import math
import traceback
import numpy as np
import torch
from torch import nn
import experiment as exp
import worker as old
from exact_refinement import ExactResponseNetwork
from short_train import TrialSystem
from fp32_response import make_typed_optics
from trajectory_backward import four_steps

SOURCE = ROOT/'nikon_large_debug_20261025/trajectory_4ep/candidate.pt'
PACKET_SOURCE = ROOT/'nikon_abs_three_radii_10ep_20261025'
EPOCHS, BATCH, SEED = 4, 128, 2026102501
EXTRA_FIELDS = (-.1, .1, -.15, .15)
EXTRA_DIM = 197+4*49


def state(status, **kw):
    exp.atomic(HERE/'state.json', dict(status=status, pid=os.getpid(), updated=exp.stamp(), **kw))


class AdditiveNetwork(ExactResponseNetwork):
    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def forward(self, image, raw=None, extra=None):
        if raw is None or raw.shape[1:] != (3,49):
            raise ValueError('Original DIFF input must have shape [N,3,49]')
        if extra is None or extra.shape != (len(raw), EXTRA_DIM):
            raise ValueError('Extra input must have shape [N,393]: center ABS197 + four DIFF49')
        prediction = exp.LargerNetwork.forward(self, image, raw)
        correction = self.extra_branch((extra-self.extra_mean)/self.extra_std)
        prior = ((prediction+correction)*self.base.target_std+self.base.target_mean).clamp(-1.5,1.5).double()
        # Same147 DIFF residual, four steps, ridge/cap/gains and corrected VJP.
        physical = four_steps(prior, raw.flatten(1).double()/7., self)
        return ((physical-self.base.target_mean)/self.base.target_std).float()


class AdditiveSystem(TrialSystem):
    def forward(self, model, data, indices):
        image = torch.as_tensor(data['image'][indices].astype(np.float32), device='cuda')
        raw = torch.as_tensor(data['raw'][indices], device='cuda')
        extra = torch.as_tensor(np.array(data['extra'][indices], copy=True), device='cuda')
        target = torch.as_tensor(data['q'][indices], dtype=torch.float64, device='cuda')
        prediction = model(image,raw,extra)
        command = -(prediction*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5)
        pose = nn.functional.smooth_l1_loss(prediction,
            ((target-model.base.target_mean)/model.base.target_std).float(), beta=.25)
        return target+command.double(), target, command, pose
    # TrialSystem.loss and System.metrics are inherited unchanged.


def prepare_extra(system, data):
    directory = HERE/'data'
    directory.mkdir(exist_ok=True)
    ready, path = directory/'ready.json', directory/'extra.npy'
    if ready.exists():
        return np.load(path,mmap_mode='r')
    with np.load(PACKET_SOURCE/'data/metadata.npz') as z:
        lookup = {int(identity):i for i,identity in enumerate(z['ids'])}
    source_rows = np.asarray([lookup[int(identity)] for identity in data['ids']])
    center_source = np.load(PACKET_SOURCE/'data/raw.npy',mmap_mode='r')
    marker = directory/'generation.json'
    start = json.loads(marker.read_text())['completed'] if marker.exists() else 0
    extra = np.lib.format.open_memmap(path,mode='r+' if start else 'w+',
                                    dtype=np.float32,shape=(len(data['ids']),EXTRA_DIM))
    optics = system.legacy.NikonTensorOptics(nodes=system.legacy.nikon35.NODES,
                                           fields=EXTRA_FIELDS,device='cuda')
    optics = make_typed_optics(optics,torch.float64)
    tick = exp.active_time()
    for first in range(start,len(extra),256):
        last = min(first+256,len(extra))
        with torch.no_grad():
            q = torch.as_tensor(data['q'][first:last],dtype=torch.float64,device='cuda')
            traced = optics.trace(q+optics.bias)
            signal = ((traced['opl'][:,:,1:]-optics.nominal_opl)@optics.projector.T/optics.wave)
            valid = traced['valid'].all(-1).all(-1) & torch.isfinite(signal).all(-1).all(-1)
            if not bool(valid.all()):
                raise RuntimeError('Invalid added observation; no case is silently filtered')
            values = signal.cpu().numpy()
        for row,identity in enumerate(data['ids'][first:last]):
            rng = np.random.default_rng(np.random.SeedSequence([2026102602,int(identity),496]))
            values[row] += rng.normal(0.,.015,values[row].shape)
        extra[first:last,:197] = center_source[source_rows[first:last],0]
        extra[first:last,197:] = values.reshape(last-first,-1).astype(np.float32)
        extra.flush()
        exp.atomic(marker,dict(completed=last,total=len(extra)))
        state('preparing_added_observations',completed=last,total=len(extra),active_seconds=exp.active_time()-tick)
        if (HERE/'STOP').exists():
            state('paused',stage='preparing_added_observations',completed=last)
            return None
    exp.atomic(ready,dict(count=len(extra),extra_dim=EXTRA_DIM,
        original_observations='Original stored maps and147 DIFF values reused without resampling or noise redraw',
        center_abs='197 values reused by case ID from existing same-initial-pose cache',
        new_diff_fields_mm=list(EXTRA_FIELDS),new_diff_nodes_per_field=49,noise_std_waves=.015,
        measurement_acquisitions=1,pose_used_only_for_simulation_and_training_labels=True,
        independent_review=False))
    return extra


def make_model(system, stats):
    model, _ = old.make_model(system)
    saved = torch.load(SOURCE,map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True)
    model.__class__ = AdditiveNetwork
    model.register_buffer('extra_mean',torch.tensor(stats['mean'],dtype=torch.float32,device='cuda'))
    model.register_buffer('extra_std',torch.tensor(stats['std'],dtype=torch.float32,device='cuda'))
    model.extra_branch = nn.Sequential(nn.Linear(EXTRA_DIM,256),nn.GELU(),
        nn.Linear(256,256),nn.GELU(),nn.Linear(256,35)).to(device='cuda',dtype=torch.float32)
    nn.init.zeros_(model.extra_branch[-1].weight)
    nn.init.zeros_(model.extra_branch[-1].bias)
    model.requires_grad_(True)
    return model


def fixed_sample(data):
    selected = json.loads((PACKET_SOURCE/'pause_sample100/selection.json').read_text())
    lookup = {int(identity):i for i,identity in enumerate(data['ids'])}
    indices = np.asarray([lookup[int(identity)] for identity in selected['original_ids']])
    if len(indices)!=100 or bool(data['train'][indices].any()):
        raise RuntimeError('Expected the previously selected100 original validation cases')
    exp.atomic(HERE/'sample_selection.json',dict(indices=indices.tolist(),
        original_ids=selected['original_ids'],source=str(PACKET_SOURCE/'pause_sample100/selection.json'),
        changed_after_observing_results=False,count=100,training_rows=0))
    return indices


def main():
    import msvcrt
    lock = (HERE/'training.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (HERE/'summary.json').exists():
        return
    if (HERE/'STOP').exists():
        state('paused',reason='Local STOP');return
    if os.name=='nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        state('loading_original_data')
        system = old.make_system('nikon')
        system.__class__ = AdditiveSystem
        data = old.load_data(system)
        data['extra'] = prepare_extra(system,data)
        if data['extra'] is None:
            return
        train_ids = np.flatnonzero(data['train'])
        sample_ids = fixed_sample(data)
        stats_path = HERE/'data/extra_normalization.npz'
        if stats_path.exists():
            with np.load(stats_path) as z:
                stats = {k:z[k].copy() for k in z.files}
        else:
            values = np.asarray(data['extra'][train_ids],dtype=np.float64)
            stats = dict(mean=values.mean(0).astype(np.float32),
                         std=np.maximum(values.std(0),.015).astype(np.float32))
            np.savez(stats_path,**stats);del values
        state('building_additive_model')
        model = make_model(system,stats)
        optimizer = exp.make_optimizer(model)
        config = dict(system='nikon',trial='additive_observations_only',epochs=EPOCHS,batch=BATCH,seed=SEED,
            parameters=sum(p.numel() for p in model.parameters()),
            extra_branch_parameters=sum(p.numel() for p in model.extra_branch.parameters()),
            neural_dtype='float32',source_checkpoint=str(SOURCE),source_epoch=3,
            training_count=28000,original_training_count=24000,all4000_added_in_training=True,
            validation_population=6000,evaluation_count=100,evaluation_epochs=[0,4],
            measurement_acquisitions=1,measured_feedback=False,model_refinements=4,
            native_input_shape=[3,49],native_fields_mm=[0.,-.05,.05],native_input_kind='DIFF',
            extra_inputs=dict(center_abs197=True,edge_diff49_fields_mm=list(EXTRA_FIELDS),count=EXTRA_DIM),
            total_measured_scalars=147+EXTRA_DIM,original_preprocessing_preserved=True,
            original_architecture_and_heads_preserved=True,new_branch_initial_output=0.,
            all_neural_parameters_trainable=True,bn_running_statistics='Frozen as in the original repaired run',
            inverse_residual='Unchanged original147 DIFF; original ridge1e-4, cap0.25 and gains',
            training_backward='Existing corrected four-step trajectory VJP, unchanged',
            loss='Inherited original TrialSystem exact_local_response loss: centerABS +/-0.1 DIFF on197 nodes',
            initial_learning_rates=[1e-5,1e-4],cosine_horizon=40,weight_decay=1e-4,gradient_clip=5.,
            no_independent_review=True,no_full_validation=True,no_per_epoch_validation=True,
            followup_training=False,gpu=torch.cuda.get_device_name())
        exp.atomic(HERE/'config.json',config)
        epoch,next_batch,total_seconds = 1,0,0.
        loss_sum,processed,epoch_seconds = 0.,0,0.
        history,baseline = [],None
        if (HERE/'resume.pt').exists():
            saved = torch.load(HERE/'resume.pt',map_location='cpu',weights_only=False)
            if saved['config'] != config:
                raise RuntimeError('Resume config differs')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            epoch,next_batch,total_seconds,loss_sum,processed,epoch_seconds,history,baseline = [
                saved[k] for k in ('epoch','next_batch','total_seconds','loss_sum','processed',
                                  'epoch_seconds','history','baseline')]
            torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
            del saved
        def checkpoint():
            exp.save_pt(HERE/'resume.pt',dict(config=config,model=model.state_dict(),optimizer=optimizer.state_dict(),
                epoch=epoch,next_batch=next_batch,total_seconds=total_seconds,loss_sum=loss_sum,
                processed=processed,epoch_seconds=epoch_seconds,history=history,baseline=baseline,
                rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        if baseline is None:
            state('baseline_sample100',epoch=0,epochs=EPOCHS)
            baseline = exp.evaluate(system,model,data,sample_ids,BATCH,HERE/'baseline_sample100.npz')
            exp.atomic(HERE/'baseline_sample100.json',baseline)
            print(json.dumps(dict(stage='baseline_sample100',result=baseline)),flush=True)
            checkpoint()
        batches = math.ceil(len(train_ids)/BATCH)
        while epoch<=EPOCHS:
            order = np.random.default_rng(SEED+epoch).permutation(train_ids)
            factor = .1+.9*.5*(1+math.cos(math.pi*(epoch-1)/40))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*factor
            for b in range(next_batch,batches):
                if (HERE/'STOP').exists():
                    checkpoint();state('paused',epoch=epoch,batch=next_batch,epochs=EPOCHS);return
                tick = exp.active_time()
                model.train();optimizer.zero_grad(set_to_none=True)
                indices = order[b*BATCH:(b+1)*BATCH]
                loss,parts = system.loss(model,data,indices)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f'Nonfinite loss: {parts}')
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step();torch.cuda.synchronize()
                seconds = exp.active_time()-tick
                total_seconds += seconds;epoch_seconds += seconds
                loss_sum += float(loss.detach())*len(indices);processed += len(indices);next_batch=b+1
                if next_batch%25==0 or next_batch==batches:
                    checkpoint()
                if next_batch%5==0 or next_batch==1 or next_batch==batches:
                    updates=(epoch-1)*batches+next_batch
                    state('training',epoch=epoch,epochs=EPOCHS,batch=next_batch,batches=batches,
                        optimizer_updates=updates,total_optimizer_updates=EPOCHS*batches,
                        loss=loss_sum/processed,last_parts=parts,gradient_norm=float(grad),
                        active_training_seconds=total_seconds,seconds_per_batch=total_seconds/updates,
                        remaining_training_seconds=total_seconds/updates*(EPOCHS*batches-updates),
                        baseline_sample100_passes=baseline['full_field_passes'],
                        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                    print(json.dumps(dict(epoch=epoch,batch=next_batch,loss=loss_sum/processed,
                                          seconds_per_batch=total_seconds/updates)),flush=True)
            history.append(dict(epoch=epoch,loss=loss_sum/processed,active_seconds=epoch_seconds))
            exp.atomic(HERE/'history.json',history)
            epoch+=1;next_batch=0;loss_sum=0.;processed=0;epoch_seconds=0.;checkpoint()
        exp.save_pt(HERE/'last.pt',dict(config=config,model=model.state_dict(),epoch=EPOCHS))
        if (HERE/'STOP').exists():
            state('paused',epoch=EPOCHS,stage='before_final_sample100');return
        state('final_sample100',epoch=EPOCHS,epochs=EPOCHS)
        final = exp.evaluate(system,model,data,sample_ids,BATCH,HERE/'final_sample100.npz')
        exp.atomic(HERE/'final_sample100.json',final)
        result = dict(status='completed',epochs=EPOCHS,optimizer_updates=EPOCHS*batches,
            baseline_sample100=baseline,final_sample100=final,active_training_seconds=total_seconds,
            recovery_change_percentage_points=100*(final['full_field_rate']-baseline['full_field_rate']),
            samples=100,full_validation=False,independent_review=False,automatic_deployment=False,
            assistant_interpretation='pending',updated=exp.stamp())
        exp.atomic(HERE/'summary.json',result)
        state('completed',epoch=EPOCHS,epochs=EPOCHS,baseline_passes=baseline['full_field_passes'],
              final_passes=final['full_field_passes'])
        print(json.dumps(result),flush=True)
    except BaseException:
        state('failed',error=traceback.format_exc())
        raise
    finally:
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':
    main()
