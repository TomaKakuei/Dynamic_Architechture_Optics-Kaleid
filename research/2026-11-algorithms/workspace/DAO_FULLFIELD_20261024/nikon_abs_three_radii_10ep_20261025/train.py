"""Authorized Nikon-only ten-epoch run: one seven-field observation packet.

No replay, gradient audit, confirmation experiment, or per-epoch validation.
One ordinary validation after epoch 10; historical runs remain paused.
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
from scipy.spatial import Delaunay, cKDTree
import experiment as exp
import worker as old
from fp32_response import make_typed_optics
from trajectory_backward import four_steps

FIELDS = (0., -.05, .05, -.1, .1, -.15, .15)
CHANNELS = ['center_ABS', 'minus_0p05_DIFF', 'plus_0p05_DIFF',
            'minus_0p1_DIFF', 'plus_0p1_DIFF', 'minus_0p15_DIFF', 'plus_0p15_DIFF']
FIELD_MAP = (0, 1, 2, 1, 2, 1, 2)
FIELD_SCALE = (1., 1/3, 1/3, 1/3, 1/3, 1/3, 1/3)
N_FIELDS = len(FIELDS)
NODES = 197
INPUTS = N_FIELDS*NODES
PREVIOUS_RUN = ROOT/'nikon_abs_wide_10ep_20261025'
SOURCE = ROOT/'nikon_large_debug_20261025/trajectory_4ep/candidate.pt'
EPOCHS = 10
BATCH = 128
MICRO = 64
SEED = 2026102503
NOISE = .015


def state(status, **kw):
    exp.atomic(HERE/'state.json', dict(status=status, pid=os.getpid(), updated=exp.stamp(), **kw))


def packet(optics, physical, per_field_valid=False):
    """Same signed ABS/DIFF and PTTD definition for sensors, inverse and loss."""
    traced = optics.trace(physical.to(optics.bias.dtype) + optics.bias)
    hit = traced['hit'][:, :1, 1:]
    direction = traced['direction'][:, :1, 1:]
    direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    delta = hit - traced['hit'][:, :1, :1]
    projected = (delta*direction).sum(-1)
    radicand = projected.square() - delta.square().sum(-1) + 100.
    sphere = traced['opl'][:, :1, 1:] - projected - torch.sqrt(radicand.clamp_min(1e-30))
    absolute = sphere @ optics.projector.T / optics.wave
    difference = ((traced['opl'][:, 1:, 1:]-optics.nominal_opl[1:])
                  @ optics.projector.T / optics.wave)
    values = torch.cat((absolute, difference), dim=1)
    field_valid = traced['valid'].all(-1) & torch.isfinite(values).all(-1)
    center_valid = field_valid[:, :1] & (radicand > 0).all(-1)
    field_valid = torch.cat((center_valid, field_valid[:, 1:]), dim=1)
    return values, field_valid if per_field_valid else field_valid.all(1)


def metadata(legacy):
    directory = HERE/'data'
    directory.mkdir(exist_ok=True)
    path = directory/'metadata.npz'
    if path.exists():
        with np.load(path) as z:
            return {k: z[k].copy() for k in z.files}
    previous_metadata = PREVIOUS_RUN/'data/metadata.npz'
    if previous_metadata.exists():
        with np.load(previous_metadata) as z:
            data = {k:z[k].copy() for k in z.files}
        np.savez(path, **data)
        exp.atomic(directory/'sources.json', dict(metadata_reused_from=str(previous_metadata),
            training_count=28000, validation_count=6000, added_training_count=4000,
            all_added_in_training=True))
        return data
    rows, files = [], []
    for path, group, training, _ in legacy.sources():
        with np.load(path) as z:
            ids = z['seeds'].copy()
            rows.append((z['targets'].astype(np.float64), ids,
                         np.full(len(ids), training, dtype=bool), np.full(len(ids), group)))
        files.append(str(path))
    for path in sorted((ROOT/'original_routes_40ep_20261025/nikon_data_v2').glob('block_*.npz')):
        with np.load(path) as z:
            rows.append(tuple(z[k].copy() for k in ('q', 'ids', 'train', 'cohort')))
        files.append(str(path))
    data = dict(zip(('q', 'ids', 'train', 'cohort'),
                    (np.concatenate([row[k] for row in rows]) for k in range(4))))
    if len(data['q']) != 34000 or int(data['train'].sum()) != 28000:
        raise RuntimeError('Expected 24000 original + 4000 added training, 6000 validation')
    np.savez(directory/'metadata.npz', **data)
    exp.atomic(directory/'sources.json', dict(files=files, training_count=28000,
        validation_count=6000, added_training_count=4000, all_added_in_training=True))
    return data


def prepare_observations(optics, data, legacy):
    directory = HERE/'data'
    ready = directory/'ready.json'
    path = directory/'raw.npy'
    if ready.exists():
        return np.load(path, mmap_mode='r')
    marker = directory/'generation.json'
    start = json.loads(marker.read_text())['completed'] if marker.exists() else 0
    raw = np.lib.format.open_memmap(path, mode='r+' if start else 'w+',
                                  dtype=np.float32, shape=(len(data['q']), N_FIELDS, NODES))
    # Existing center / +/-0.1 / +/-0.15 readings are at the SAME initial poses.
    # Reuse them exactly; compute only the missing +/-0.05 pair once.
    existing = np.load(PREVIOUS_RUN/'data/raw.npy', mmap_mode='r')
    pair_optics = legacy.NikonTensorOptics(nodes=optics.nodes, fields=(-.05, .05), device='cuda')
    pair_optics = make_typed_optics(pair_optics, torch.float64)
    tick = exp.active_time()
    initial_start = start
    for first in range(start, len(raw), 256):
        last = min(first+256, len(raw))
        with torch.no_grad():
            q = torch.as_tensor(data['q'][first:last], device='cuda')
            traced = pair_optics.trace(q+pair_optics.bias)
            signal = ((traced['opl'][:,:,1:]-pair_optics.nominal_opl)
                      @ pair_optics.projector.T / pair_optics.wave)
            valid = traced['valid'].all(-1).all(-1) & torch.isfinite(signal).all(-1).all(-1)
            if not bool(valid.all()):
                bad = data['ids'][first:last][~valid.cpu().numpy()].tolist()
                raise RuntimeError(f'Invalid initial sensor packet at expanded fields; no cases discarded: {bad}')
            values = signal.cpu().numpy()
        for row, identity in enumerate(data['ids'][first:last]):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, int(identity), 705]))
            values[row] += rng.normal(0., NOISE, values[row].shape)
        raw[first:last, 0] = existing[first:last, 0]
        raw[first:last, 1:3] = values.astype(np.float32)
        raw[first:last, 3:] = existing[first:last, 1:]
        raw.flush()
        exp.atomic(marker, dict(completed=last, total=len(raw)))
        seconds = exp.active_time()-tick
        state('generating_observations', completed=last, total=len(raw),
              active_seconds=seconds, remaining_seconds=seconds/max(1,last-initial_start)*(len(raw)-last))
        if (HERE/'STOP').exists():
            state('paused', stage='generating_observations', completed=last)
            return None
    exp.atomic(ready, dict(shape=list(raw.shape), fields_mm=list(FIELDS),
        channels=CHANNELS, reused_five_channels_from=str(PREVIOUS_RUN/'data/raw.npy'),
        generated_channels_only=['minus_0p05_DIFF','plus_0p05_DIFF'],
        nodes_per_field=197, noise_std_waves=NOISE, one_fixed_pose_acquisition=True,
        reference='Existing fixed corrected nominal bias; same 197-node PTTD projection',
        training_statistics_only=True, pose_labels_not_network_inputs=True))
    return raw


def interpolation(nodes):
    axis = np.linspace(-1., 1., 32)
    x, y = np.meshgrid(axis, axis)
    points = np.c_[x.ravel(), y.ravel()]
    tri = Delaunay(nodes)
    simplex = tri.find_simplex(points)
    use = np.flatnonzero((simplex >= 0) & ((points*points).sum(1) <= 1.))
    transform = tri.transform[simplex[use]]
    b = np.einsum('nij,nj->ni', transform[:, :2], points[use]-transform[:, 2])
    weights = np.c_[b, 1.-b.sum(1)]
    matrix = np.zeros((32*32, len(nodes)), dtype=np.float32)
    matrix[use[:, None], tri.simplices[simplex[use]]] = weights
    return matrix.reshape(32, 32, len(nodes))


class WidePacketNetwork(exp.LargerNetwork):
    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def difference(self, physical):
        values, _ = packet(self.native_optics, physical)
        return values.flatten(1)/math.sqrt(197)

    @torch.no_grad()
    def response(self, physical):
        def one(q):
            return self.difference(q[None])[0]
        with torch.inference_mode(False), torch.no_grad():
            normal = physical.clone()
            return torch.func.vmap(torch.func.jacfwd(one))(normal)

    def forward(self, image, raw=None):
        if raw is None or raw.shape[1:] != (N_FIELDS, NODES):
            raise ValueError('Expected [N,7,197]: ABS(0), DIFF(-.05,+.05,-.1,+.1,-.15,+.15)')
        prediction = exp.LargerNetwork.forward(self, image, raw)
        prior = (prediction*self.base.target_std+self.base.target_mean).clamp(-1.5, 1.5).double()
        physical = four_steps(prior, raw.flatten(1).double()/math.sqrt(197), self)
        return ((physical-self.base.target_mean)/self.base.target_std).float()

    def predict_packet(self, raw):
        raw = torch.as_tensor(raw, device=self.field_identity.device, dtype=torch.float32)
        image = torch.einsum('nfk,hwk->nfhw', raw, self.grid_matrix)
        prediction = self(image, raw)
        return -(prediction*self.base.target_std+self.base.target_mean).clamp(-1.5, 1.5)


def replace_input_convs(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d) and child.in_channels == 3:
            new = nn.Conv2d(N_FIELDS, child.out_channels, child.kernel_size, child.stride,
                            child.padding, child.dilation, child.groups, child.bias is not None,
                            child.padding_mode, device='cuda', dtype=torch.float32)
            with torch.no_grad():
                new.weight.copy_(child.weight[:, FIELD_MAP] *
                                 child.weight.new_tensor(FIELD_SCALE)[None,:,None,None])
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            setattr(module, name, new)
        else:
            replace_input_convs(child)


def expanded_raw_layer(layer, old_nodes, new_nodes):
    nearest = cKDTree(new_nodes).query(old_nodes)[1]
    new = nn.Linear(INPUTS, layer.out_features, device='cuda', dtype=torch.float32)
    with torch.no_grad():
        new.weight.zero_()
        old_weight = layer.weight.reshape(layer.out_features, 3, 49)
        for field, source in enumerate(FIELD_MAP):
            indices = torch.as_tensor(nearest+field*197, device='cuda')
            new.weight.index_add_(1, indices, old_weight[:, source]*FIELD_SCALE[field])
        new.bias.copy_(layer.bias)
    return new


def make_model(system, optics, stats):
    model, _ = old.make_model(system)
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'], strict=True)
    model.__class__ = WidePacketNetwork
    model.native_optics = optics
    nodes = np.asarray(optics.nodes)
    old_nodes = np.asarray(system.legacy.nikon35.NODES)
    for network in model.base.base.networks:
        replace_input_convs(network.stem)
    model.base.raw_global[0] = expanded_raw_layer(model.base.raw_global[0], old_nodes, nodes)
    model.raw_branch[0] = expanded_raw_layer(model.raw_branch[0], old_nodes, nodes)
    model.base.raw_mean = torch.as_tensor(stats['mean'], device='cuda', dtype=torch.float32)
    model.base.raw_std = torch.as_tensor(stats['std'], device='cuda', dtype=torch.float32)
    xy = np.tile(nodes, (N_FIELDS,1))
    coords = np.c_[xy, np.repeat(np.asarray(FIELDS)/.15, 197), (xy*xy).sum(1)]
    model.base.node_coordinates = torch.tensor(coords, device='cuda', dtype=torch.float32)
    mapping = torch.tensor(FIELD_MAP, device='cuda')
    model.field_identity = nn.Parameter(model.field_identity[:, mapping].detach().clone())
    norm = nn.LayerNorm(N_FIELDS*256, device='cuda')
    linear = nn.Linear(N_FIELDS*256, 512, device='cuda')
    with torch.no_grad():
        norm.weight.copy_(model.head[0].weight.reshape(3,256)[mapping].flatten())
        norm.bias.copy_(model.head[0].bias.reshape(3,256)[mapping].flatten())
        w = model.head[1].weight.reshape(512,3,256)[:, mapping]
        w = w*w.new_tensor(FIELD_SCALE)[None,:,None]
        linear.weight.copy_(w.flatten(1)); linear.bias.copy_(model.head[1].bias)
    model.head[0], model.head[1] = norm, linear
    model.register_buffer('grid_matrix', torch.as_tensor(interpolation(nodes), device='cuda'))
    model.requires_grad_(True)
    return model


def training_loss(model, raw, targets):
    command = model.predict_packet(raw)
    target = torch.as_tensor(targets, dtype=torch.float64, device='cuda')
    endpoint = target+command.double()
    with torch.no_grad():
        _, valid = packet(model.native_optics, endpoint)
    if bool(valid.any()):
        signals, _ = packet(model.native_optics, endpoint[valid])
        fields = signals.square().mean(-1).clamp_min(1e-20).sqrt()
        ell = torch.log1p((fields/.07).square()); term = ell/(1+ell)
        physical = ((.5*term.mean(1)+.5*term.amax(1)).sum()+100*(~valid).sum())/len(target)
    else:
        physical = command.sum()*0.+100.
    inferred = -command
    pose = nn.functional.smooth_l1_loss(
        (inferred-model.base.target_mean)/model.base.target_std,
        ((target-model.base.target_mean)/model.base.target_std).float(), beta=.25)
    return physical+.02*pose, int((~valid).sum())


@torch.no_grad()
def final_validation(model, data, raw):
    model.eval()
    indices = np.flatnonzero(~data['train'])
    fields, valids, commands = [], [], []
    tick = exp.active_time()
    for start in range(0, len(indices), MICRO):
        ix = indices[start:start+MICRO]
        command = model.predict_packet(np.array(raw[ix], copy=True))
        signal, valid = packet(model.native_optics,
                               torch.as_tensor(data['q'][ix], device='cuda')+command.double(),
                               per_field_valid=True)
        fields.append(signal.square().mean(-1).sqrt().cpu().numpy())
        valids.append(valid.cpu().numpy()); commands.append(command.cpu().numpy())
        state('final_validation', completed=min(start+MICRO,len(indices)), total=len(indices),
              epoch=EPOCHS, epochs=EPOCHS, active_seconds=exp.active_time()-tick)
    w, field_valid, command = [np.concatenate(v) for v in (fields, valids, commands)]
    valid = field_valid.all(1)
    def score(columns):
        selected = w[:, columns]
        passing = field_valid[:, columns].all(1) & np.isfinite(selected).all(1) & (selected.max(1)<.07)
        return dict(passes=int(passing.sum()), count=len(w), recovery_rate=float(passing.mean()))
    result = dict(epoch=EPOCHS, seven_fields=score(list(range(N_FIELDS))),
        original_three_fields=score([0,3,4]),
        center_plus_each_radius={str(radius):score([0,2*i+1,2*i+2])
                                 for i,radius in enumerate((.05,.1,.15))},
        center_passes=int((field_valid[:,0] & (w[:,0]<.07)).sum()), invalid=int((~valid).sum()),
        fields_mm=list(FIELDS), per_field_mean=w[valid].mean(0).tolist() if valid.any() else None,
        per_field_passes=((w<.07)&field_valid).sum(0).tolist(), active_seconds=exp.active_time()-tick,
        independent_review_performed=False)
    np.savez(HERE/'final_validation.npz', original_ids=data['ids'][indices], cohort=data['cohort'][indices],
             initial=data['q'][indices], command=command, fields=w, valid=valid, field_valid=field_valid)
    exp.atomic(HERE/'final_validation.json', result)
    return result


def main():
    import msvcrt
    lock = (HERE/'training.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    if (HERE/'summary.json').exists():
        return
    if (HERE/'STOP').exists():
        state('paused', reason='Local STOP marker'); return
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        state('loading_metadata')
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        system = old.make_system('nikon')
        optics = system.legacy.NikonTensorOptics(nodes=system.legacy.nikon35.AUDIT_NODES,
                                               fields=FIELDS, device='cuda')
        optics = make_typed_optics(optics, torch.float64)
        data = metadata(system.legacy)
        raw = prepare_observations(optics, data, system.legacy)
        if raw is None:
            return
        train_ids = np.flatnonzero(data['train'])
        stats_path = HERE/'data/normalization.npz'
        if stats_path.exists():
            with np.load(stats_path) as z:
                stats = {k:z[k].copy() for k in z.files}
        else:
            training = np.asarray(raw[train_ids], dtype=np.float64)
            stats = dict(mean=training.mean(0).astype(np.float32),
                         std=np.maximum(training.std(0), NOISE).astype(np.float32))
            np.savez(stats_path, **stats); del training
        state('building_network')
        model = make_model(system, optics, stats)
        del system; gc.collect(); torch.cuda.empty_cache()
        optimizer = exp.make_optimizer(model)
        config = dict(system='nikon', epochs=EPOCHS, seed=SEED, batch=BATCH, microbatch=MICRO,
            training_count=28000, validation_count=6000, all_4000_added_points_in_training=True,
            fields_mm=list(FIELDS), field_kinds=['ABS']+['DIFF']*6, channels=CHANNELS, nodes_per_field=197,
            input_shape=[N_FIELDS,NODES], measurement_acquisitions=1, measured_feedback=False,
            model_refinements=4, parameters=sum(p.numel() for p in model.parameters()),
            neural_dtype='float32', source_checkpoint=str(SOURCE), source_epoch=3,
            architecture='Original 15/15/5 grouped CNNs plus 4-layer wide field attention; seven-field inputs',
            bn_running_statistics='Frozen; affine and all neural parameters trainable',
            input_projection='Shared 197-node PTTD; same packet function for data, inverse and endpoint loss',
            initialization='Reuse E3 weights; input interfaces expanded, not function-preserving',
            loss='Original bounded .5mean+.5worst on all seven fields + .02 pose; invalid100',
            initial_learning_rates=[1e-5,1e-4], scheduler='Cosine decay over 10 epochs',
            ordinary_validation='Once after epoch10 only; original3, all7 and each radius from same trace',
            baseline_validation=False, per_epoch_validation=False, independent_review=False,
            derivative_checks=False, measured_closed_loop=False, gpu=torch.cuda.get_device_name())
        exp.atomic(HERE/'config.json', config)
        epoch, next_batch, total_seconds = 1, 0, 0.
        total_loss, processed, invalid_count, epoch_seconds = 0., 0, 0, 0.
        history = []
        if (HERE/'resume.pt').exists():
            saved = torch.load(HERE/'resume.pt', map_location='cpu', weights_only=False)
            if saved['config'] != config:
                raise RuntimeError('Resume configuration differs')
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            epoch, next_batch, total_seconds, total_loss, processed, invalid_count, epoch_seconds, history = [
                saved[k] for k in ('epoch','next_batch','total_seconds','total_loss','processed',
                                  'invalid_count','epoch_seconds','history')]
            torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
            del saved
        def checkpoint():
            exp.save_pt(HERE/'resume.pt', dict(config=config, model=model.state_dict(),
                optimizer=optimizer.state_dict(), epoch=epoch, next_batch=next_batch,
                total_seconds=total_seconds, total_loss=total_loss, processed=processed,
                invalid_count=invalid_count, epoch_seconds=epoch_seconds, history=history,
                rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
        batches = math.ceil(len(train_ids)/BATCH)
        while epoch <= EPOCHS:
            order = np.random.default_rng(SEED+epoch).permutation(train_ids)
            factor = .1+.9*.5*(1+math.cos(math.pi*(epoch-1)/EPOCHS))
            for group in optimizer.param_groups:
                group['lr'] = group['initial_lr']*factor
            for b in range(next_batch, batches):
                if (HERE/'STOP').exists():
                    checkpoint(); state('paused',epoch=epoch,batch=next_batch,epochs=EPOCHS); return
                tick = exp.active_time()
                model.train(); optimizer.zero_grad(set_to_none=True)
                indices = order[b*BATCH:(b+1)*BATCH]
                batch_loss, batch_invalid = 0., 0
                for start in range(0,len(indices),MICRO):
                    ix = indices[start:start+MICRO]
                    loss, invalid = training_loss(model, np.array(raw[ix],copy=True), data['q'][ix])
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError('Nonfinite training loss')
                    (loss*(len(ix)/len(indices))).backward()
                    batch_loss += float(loss.detach())*len(ix); batch_invalid += invalid
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step(); torch.cuda.synchronize()
                elapsed = exp.active_time()-tick
                total_seconds += elapsed; epoch_seconds += elapsed
                total_loss += batch_loss; processed += len(indices); invalid_count += batch_invalid
                next_batch = b+1
                if next_batch % 25 == 0 or next_batch == batches:
                    checkpoint()
                if next_batch % 5 == 0 or next_batch == 1 or next_batch == batches:
                    completed = (epoch-1)*batches+next_batch
                    state('training',epoch=epoch,epochs=EPOCHS,batch=next_batch,batches=batches,
                        optimizer_updates=completed,total_optimizer_updates=EPOCHS*batches,
                        loss=total_loss/processed,invalid=invalid_count,gradient_norm=float(grad),
                        active_training_seconds=total_seconds,seconds_per_batch=total_seconds/completed,
                        remaining_training_seconds=total_seconds/completed*(EPOCHS*batches-completed),
                        peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                    print(json.dumps(dict(epoch=epoch,batch=next_batch,loss=total_loss/processed,
                                          seconds_per_batch=total_seconds/completed)),flush=True)
            history.append(dict(epoch=epoch,loss=total_loss/processed,invalid=invalid_count,
                                active_seconds=epoch_seconds))
            exp.atomic(HERE/'history.json', history)
            epoch += 1; next_batch = 0; total_loss = 0.; processed = 0; invalid_count = 0; epoch_seconds = 0.
            checkpoint()
        exp.save_pt(HERE/'last.pt', dict(model=model.state_dict(),config=config,epoch=EPOCHS))
        if (HERE/'STOP').exists():
            state('paused',stage='before_final_validation',epoch=EPOCHS,epochs=EPOCHS); return
        result = final_validation(model,data,raw)
        exp.atomic(HERE/'summary.json', dict(status='completed',epochs=EPOCHS,
            optimizer_updates=EPOCHS*batches,active_training_seconds=total_seconds,
            validation=result,independent_review_performed=False,assistant_interpretation='pending',
            updated=exp.stamp()))
        state('completed',epoch=EPOCHS,epochs=EPOCHS,validation=result)
    except BaseException:
        state('failed',error=traceback.format_exc())
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == '__main__':
    main()
