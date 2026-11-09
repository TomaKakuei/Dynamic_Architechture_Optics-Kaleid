"""Original two DAO routes, extra training data, function-preserving 2x widening."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(PARENT/'nikon_short_training'))
import argparse
import ctypes
import gc
import json
import math
import multiprocessing as mp
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torch
from torch import nn
import experiment as exp
from durable_io import atomic_json as atomic

WIDTH = {'nikon': 4496, 'trepan': 4632}
SOURCE = {'nikon': PARENT/'nikon_short_training/exact_local_response/candidate.pt',
          'trepan': PARENT/'trepan_large/candidate.pt'}


def progress(path, status, **kw):
    atomic(path, dict(status=status, pid=os.getpid(), updated=exp.stamp(), **kw))


def make_system(name):
    if name == 'nikon':
        from short_train import TrialSystem
        system = TrialSystem('exact_local_response')
        system.warm = SOURCE[name]
        saved = torch.load(system.warm, map_location='cpu', weights_only=False)
        # TrialSystem attaches native_optics after loading the neural submodule.
        # The complete optical buffers are restored by make_model afterward.
        system.saved = dict(saved, model={k:v for k,v in saved['model'].items()
                                         if not k.startswith('native_optics.')})
    else:
        system = exp.System(name)
    return system


def make_model(system, verify=False):
    model = system.model('large')
    saved = torch.load(SOURCE[system.name], map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    before_count = sum(p.numel() for p in model.parameters())
    model.eval()
    if verify:
        data = system.load_data(pilot=True)
        image = torch.tensor(data['image'][:2].astype(np.float32), device='cuda')
        raw = None if data['raw'] is None else torch.tensor(data['raw'][:2], device='cuda')
        with torch.no_grad():
            before = exp.LargerNetwork.forward(model, image, raw).clone()
        del data
    for layer in model.layers:
        old1, old2 = layer.linear1, layer.linear2
        width = WIDTH[system.name]
        new1 = nn.Linear(old1.in_features, width, device='cuda', dtype=torch.float32)
        new2 = nn.Linear(width, old2.out_features, device='cuda', dtype=torch.float32)
        with torch.no_grad():
            new1.weight[:old1.out_features].copy_(old1.weight)
            new1.bias[:old1.out_features].copy_(old1.bias)
            new2.weight.zero_()
            new2.weight[:, :old2.in_features].copy_(old2.weight)
            new2.bias.copy_(old2.bias)
        layer.linear1, layer.linear2 = new1, new2
    model.requires_grad_(True)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    if verify:
        with torch.no_grad():
            after = exp.LargerNetwork.forward(model, image, raw)
        difference = float((before-after).abs().max())
        torch.testing.assert_close(after, before, rtol=2e-5, atol=2e-5)
        atomic(HERE/f'{system.name}_expansion_check.json', dict(
            status='passed', samples=2, old_parameters=before_count,
            new_parameters=sum(p.numel() for p in model.parameters()),
            neural_prediction_max_abs_difference=difference, source_epoch=int(saved['epoch']),
            source=str(SOURCE[system.name]), original_group_network_retained=True,
            original_complete_prediction_retained=True, parameter_dtype='float32',
            discarded_optimizer_updates=0, ffn_width=WIDTH[system.name]))
    return model, saved


def generate_nikon():
    out = HERE/'nikon_data_v2'; out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        return
    system = make_system('nikon')
    torch.manual_seed(2026102501)
    model, _ = make_model(system, verify=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    strata = [(2000, list(range(7))), (1000, list(range(6))),
              (500, [0, 1, 2, 6]), (500, [3, 4, 5, 6])]
    ids = np.arange(3100000000, 3100004000, dtype=np.int64)
    old_ids = []
    for path, *_ in system.legacy.sources():
        with np.load(path) as z:
            old_ids.append(z['seeds'].copy())
    assert not len(np.intersect1d(np.concatenate(old_ids), ids)), 'New measurement seeds must be disjoint'
    # Separate deterministic pose and measurement-noise streams.
    q = np.zeros((4000, 7, 5), dtype=np.float32)
    cohort = np.empty(4000, dtype=np.int64)
    offset = 0
    for group, (count, bodies) in enumerate(strata):
        for index in range(offset, offset+count):
            rng = np.random.default_rng(np.random.SeedSequence([20261025, index, 35]))
            q[index, bodies] = rng.uniform(-1, 1, (len(bodies), 5))
        cohort[offset:offset+count] = group+3
        offset += count
    q = q.reshape(4000, 35)
    tick = exp.active_time()
    for start in range(0, 4000, 100):
        path = out/f'block_{start//100:03d}.npz'
        if path.exists():
            with np.load(path) as z:
                assert np.array_equal(z['ids'], ids[start:start+100])
                assert z['raw'].shape == (100, 3, 49)
            continue
        with torch.no_grad():
            metrics = system.native_optics.metrics(torch.tensor(q[start:start+100], dtype=torch.float64, device='cuda'))
            raw = metrics['difference'].cpu().numpy()
            assert bool(metrics['valid'].all()) and np.isfinite(raw).all()
        for i, seed in enumerate(ids[start:start+100]):
            raw[i] += np.random.RandomState(int(seed)).normal(0, .015, raw[i].shape)
        maps = system.legacy.nikon35.grid(raw).astype(np.float32)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, image=maps, raw=raw.astype(np.float32),
                q=q[start:start+100].astype(np.float64), ids=ids[start:start+100],
                cohort=cohort[start:start+100], train=np.ones(100, dtype=bool))
        os.replace(temporary, path)
        progress(out/'state.json', 'generating', completed=start+100, total=4000,
                 active_seconds=exp.active_time()-tick)
    atomic(out/'manifest.json', dict(count=4000, training_count=4000, validation_count=0,
        strata=[dict(count=n, active_bodies=b) for n,b in strata], noise_std_waves=.015,
        native_observation='Original 3x49 DIFF; original 32x32 interpolation',
        new_seed_range=[int(ids[0]), int(ids[-1])], optical_merit_filter=False))
    progress(out/'state.json', 'completed', completed=4000, total=4000)


def generate_trepan():
    out = HERE/'trepan_data'; out.mkdir(exist_ok=True)
    if (out/'manifest.json').exists():
        return
    source = exp.ROOT/'trepan_restored_joint_20261008'
    sys.path.insert(0, str(source))
    import data as original
    if not (HERE/'trepan_expansion_check.json').exists():
        system = make_system('trepan')
        torch.manual_seed(2026102502)
        model, _ = make_model(system, verify=True)
        del model, system; gc.collect(); torch.cuda.empty_cache()
    calibration = json.loads((source/'camera_calibration.json').read_text())
    camera = original.Camera(calibration=calibration)
    # Per front/rear cohort: 1000 all-body high-amplitude + 1000 correlated.
    # New indices stay far from old 0..7499 and avoid old camera-seed collisions.
    jobs = []
    for block in range(40):
        for cohort in (0, 1):
            path = out/f'cohort{cohort}_{block:03d}.npz'
            indices = [1_000_000+4*(j//2)+1+j%2 for j in range(block*50, (block+1)*50)]
            if path.exists():
                with np.load(path) as z:
                    assert z['speckles'].shape == (50, 9, 128, 128)
                    assert np.array_equal(z['indices'], indices) and (z['cohort']==cohort).all()
            else:
                jobs.append((path, [(cohort, i) for i in indices]))
    tick = exp.active_time(); new = 0
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'), initializer=original.init_worker) as pool:
        pending = []
        for job_index, (path, tasks) in enumerate(jobs):
            pending.append((path, [pool.submit(original.produce, task) for task in tasks]))
            if len(pending)<2 and job_index != len(jobs)-1:
                continue
            while pending:
                dest, futures = pending.pop(0); rows=[]; images=[]
                for future in futures:
                    row = future.result()
                    seed = 91000000+row['cohort']*100000+row['index']
                    images.append(camera.capture(row['reps'], seed, calibration['grid']))
                    rows.append(row)
                original.save_block(dest, rows, images)
                new += len(rows)
                completed = 50*len(list(out.glob('cohort*.npz')))
                elapsed = exp.active_time()-tick
                progress(out/'state.json', 'generating', completed=completed, total=4000,
                    new_this_session=new, active_seconds=elapsed, seconds_per_new_sample=elapsed/new,
                    remaining_active_seconds=(4000-completed)*elapsed/new)
                print(f'Trepan data {completed}/4000, {elapsed/new:.3f} seconds/sample', flush=True)
    assert len(list(out.glob('cohort*.npz'))) == 80
    atomic(out/'manifest.json', dict(count=4000, training_count=4000, validation_count=0,
        original_sampler=True, modes={1:'all 8 bodies, amplitudes 0.4..1', 2:'correlated all-body motion'},
        per_cohort_per_mode=1000, all_bodies_move=True, optical_merit_filter=False,
        original_mechanical_checks=True, original_camera_calibration=str(source/'camera_calibration.json'),
        acquisitions_per_sample=1, new_index_min=1000001, new_index_max=1003998))
    progress(out/'state.json', 'completed', completed=4000, total=4000)


def load_data(system):
    if system.name == 'nikon':
        data = system.load_data()
        original_val_ids = data['ids'][~data['train']].copy()
        rows=[]
        for path in sorted((HERE/'nikon_data_v2').glob('block_*.npz')):
            with np.load(path) as z:
                rows.append({key:z[key].copy() for key in ('image','raw','q','ids','train','cohort')})
        assert len(rows)==40
        for key in rows[0]:
            data[key] = np.concatenate([data[key]]+[r[key] for r in rows])
        data['files'] += [str(p) for p in sorted((HERE/'nikon_data_v2').glob('block_*.npz'))]
        assert len(np.unique(data['ids']))==34000
        assert np.array_equal(data['ids'][~data['train']],original_val_ids)
        assert data['train'].sum()==28000 and (~data['train']).sum()==6000
        return data
    old_files = sorted((system.source/'data').glob('cohort*.npz'))
    new_files = sorted((HERE/'trepan_data').glob('cohort*.npz'))
    assert len(old_files)==300 and len(new_files)==80
    files=old_files+new_files
    cache=HERE/'trepan_cache';cache.mkdir(exist_ok=True)
    image_path=cache/'images.npy'
    if not (cache/'ready.json').exists():
        image=np.lib.format.open_memmap(image_path,mode='w+',dtype=np.uint16,shape=(19000,9,128,128))
        metadata=[]
        for i,path in enumerate(files):
            with np.load(path) as z:
                image[i*50:(i+1)*50]=z['speckles']
                ids=z['indices'].copy()
                training=(ids<7000) if i<300 else np.ones(50,dtype=bool)
                metadata.append((z['q'].copy(),ids,z['cohort'].copy(),training))
            if i%20==0:
                image.flush()
                progress(cache/'state.json','packing',files=i+1,total=len(files))
        image.flush();del image
        q,ids,cohort,train=[np.concatenate([r[i] for r in metadata]) for i in range(4)]
        assert len(np.unique(np.c_[cohort,ids],axis=0))==19000
        assert train.sum()==18000 and (~train).sum()==1000
        assert ((ids[~train]>=7000)&(ids[~train]<7500)).all()
        np.savez(cache/'metadata.npz',q=q,ids=ids,cohort=cohort,train=train)
        atomic(cache/'ready.json',dict(count=19000,original_count=15000,new_count=4000,files=[str(p) for p in files]))
    with np.load(cache/'metadata.npz') as z:
        data={k:z[k].copy() for k in z.files}
    data.update(image=np.load(image_path,mmap_mode='r'),raw=None,files=[str(p) for p in files])
    assert data['train'].sum()==18000 and (~data['train']).sum()==1000
    return data


def train(name):
    out=HERE/name;out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():return
    progress(out/'state.json','loading_data')
    system=make_system(name)
    data=load_data(system)
    train_ids=np.flatnonzero(data['train']);val_ids=np.flatnonzero(~data['train'])
    atomic(out/'data_manifest.json',dict(files=data['files'],original_training_count=len(train_ids)-4000,
        added_training_count=4000,training_count=len(train_ids),validation_count=len(val_ids),
        all_new_samples_in_training=True,original_validation_unchanged=True,validation_used_for_selection=True))
    seed=2026102501 if name=='nikon' else 2026102502
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model,saved=make_model(system)
    optimizer=exp.make_optimizer(model)
    batch,micro=(128,128) if name=='nikon' else (32,16)
    config=dict(system=name,epochs=40,seed=seed,batch=batch,microbatch=micro,
        parameters=sum(p.numel() for p in model.parameters()),ffn_width=WIDTH[name],
        source_checkpoint=str(SOURCE[name]),source_epoch=int(saved['epoch']),
        network_dtype='float32',all_parameters_trainable=True,original_group_network_retained=True,
        initial_learning_rates=[1e-5,1e-4],weight_decay=1e-4,gradient_clip=5.,
        scheduler='Original cosine, 1.0 to 0.1 multiplier over 40 epochs',
        original_loss_unchanged=True,measurement_acquisitions=1,measured_closed_loop=False,
        inference='Original network + 4 exact-response model-only refinements' if name=='nikon' else 'Original joint 8-body network',
        metric='Center ABS + two outer DIFF; all <0.07 waves' if name=='nikon' else 'Original full 9 fields <0.08 waves + support/mechanics',
        training_count=len(train_ids),validation_count=len(val_ids),checkpoint_every_batches=50,
        gpu=torch.cuda.get_device_name())
    del saved;gc.collect()
    epoch,next_batch,accumulated,processed,elapsed=1,0,0.,0,0.
    history=[];best=None
    if (out/'resume.pt').exists():
        resume=torch.load(out/'resume.pt',map_location='cpu',weights_only=False)
        assert resume['config']==config
        model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer'])
        epoch,next_batch,accumulated,processed,elapsed=[resume[k] for k in ('epoch','next_batch','accumulated','processed','elapsed')]
        history,best=resume['history'],resume['best']
        torch.set_rng_state(resume['rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
        del resume;gc.collect()
    atomic(out/'config.json',config)
    def payload():return dict(model=model.state_dict(),config=config,system=name,variant='original_wider')
    def checkpoint():
        exp.save_pt(out/'resume.pt',payload()|dict(optimizer=optimizer.state_dict(),epoch=epoch,
            next_batch=next_batch,accumulated=accumulated,processed=processed,elapsed=elapsed,
            history=history,best=best,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    if not history:
        progress(out/'state.json','baseline_validation',epoch=0,epochs=40)
        tick=exp.active_time()
        baseline=exp.evaluate(system,model,data,val_ids,micro,out/'baseline_validation.npz')
        history=[dict(epoch=0,validation=baseline,active_seconds=exp.active_time()-tick)]
        best=exp.selection(baseline)
        exp.save_pt(out/'candidate.pt',payload()|dict(epoch=0,validation=baseline))
        atomic(out/'selected.json',dict(epoch=0,validation=baseline))
        atomic(out/'baseline.json',baseline);exp.curves(out,history);checkpoint()
    batches=math.ceil(len(train_ids)/batch)
    while epoch<=40:
        order=np.random.default_rng(seed+epoch).permutation(train_ids)
        factor=.1+.9*.5*(1+math.cos(math.pi*(epoch-1)/40))
        for group in optimizer.param_groups:group['lr']=group['initial_lr']*factor
        for b in range(next_batch,batches):
            tick=exp.active_time();model.train();optimizer.zero_grad(set_to_none=True)
            indices=order[b*batch:(b+1)*batch];batch_loss=0.
            for start in range(0,len(indices),micro):
                ix=indices[start:start+micro]
                loss,parts=system.loss(model,data,ix)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss: '+repr(parts))
                (loss*(len(ix)/len(indices))).backward()
                batch_loss+=float(loss.detach())*len(ix)
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize()
            elapsed+=exp.active_time()-tick;accumulated+=batch_loss;processed+=len(indices);next_batch=b+1
            if next_batch%50==0 or next_batch==batches:checkpoint()
            if next_batch%10==0 or next_batch==1 or next_batch==batches:
                progress(out/'state.json','training',epoch=epoch,epochs=40,batch=next_batch,batches=batches,
                    processed=processed,loss=accumulated/processed,gradient_norm=float(grad_norm),
                    epoch_active_seconds=elapsed,training_seconds_per_batch=elapsed/next_batch,
                    remaining_training_seconds=elapsed/next_batch*(batches-next_batch+(40-epoch)*batches),
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                print(f'{name} epoch {epoch}/40 batch {next_batch}/{batches} loss {accumulated/processed:.6g}',flush=True)
        progress(out/'state.json','validation',epoch=epoch,epochs=40)
        tick=exp.active_time()
        validation=exp.evaluate(system,model,data,val_ids,micro,out/f'validation_{epoch:03d}.npz')
        elapsed+=exp.active_time()-tick
        if exp.selection(validation)<tuple(best):
            best=exp.selection(validation)
            exp.save_pt(out/'candidate.pt',payload()|dict(epoch=epoch,validation=validation))
            atomic(out/'selected.json',dict(epoch=epoch,validation=validation))
        history.append(dict(epoch=epoch,loss=accumulated/processed,validation=validation,active_seconds=elapsed))
        exp.curves(out,history)
        exp.save_pt(out/'last.pt',payload()|dict(epoch=epoch,validation=validation))
        print(json.dumps(history[-1]),flush=True)
        epoch+=1;next_batch=0;accumulated=0.;processed=0;elapsed=0.;checkpoint()
    selected=json.loads((out/'selected.json').read_text())
    selected_file=out/('baseline_validation.npz' if selected['epoch']==0 else f"validation_{selected['epoch']:03d}.npz")
    shutil.copyfile(selected_file,out/'selected_validation.npz')
    atomic(out/'summary.json',dict(status='completed',assistant_review='pending',selected_epoch=selected['epoch'],
        validation=selected['validation'],baseline=history[0]['validation'],last_validation=history[-1]['validation'],
        parameters=config['parameters'],epochs=40,optimizer_updates=40*batches,
        active_seconds=sum(h['active_seconds'] for h in history),independent_test=False,
        validation_is_selection_set=True,updated=exp.stamp()))
    progress(out/'state.json','completed',epoch=40,epochs=40,selected_epoch=selected['epoch'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['generate_nikon','generate_trepan','train_nikon','train_trepan'])
    args=parser.parse_args()
    if args.stage.endswith('_trepan'):
        raise SystemExit('Trepan is excluded by the latest user correction: Nikon only.')
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.stage.startswith('generate_'):globals()[args.stage]()
        else:train(args.stage.split('_',1)[1])
    except Exception:
        atomic(HERE/f'{args.stage}_failure.json',dict(error=traceback.format_exc(),updated=exp.stamp()))
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
