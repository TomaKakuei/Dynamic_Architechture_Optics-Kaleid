"""Reuse archived observations; full-field fine tuning with resumable checkpoints.

Each process loads exactly one optical system, avoiding historical module-name
collisions. No source data, historical checkpoints, or historical code is edited.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
import argparse
import ctypes
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch import nn
from durable_io import atomic_json, save_torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'Adaptive Optics' / 'ADAPTIVE_OPTICS_V15P'
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def stamp():
    return datetime.now(timezone.utc).isoformat()


def active_time():
    if os.name == 'nt':
        value = ctypes.c_ulonglong()
        ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value))
        return value.value / 1e7
    return time.monotonic()


def atomic(path, value):
    atomic_json(path, value)


def save_pt(path, value):
    save_torch(path, value)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LargerNetwork(nn.Module):
    """Trainable incumbent + deeper per-field CNN / 4-layer field attention.

    The zero-initialized residual head preserves the incumbent prediction at
    epoch zero. All incumbent and added parameters are subsequently trainable.
    """
    def __init__(self, base, system):
        super().__init__()
        self.base, self.system = base, system
        fields, outputs = (3, 35) if system == 'nikon' else (9, 40)
        layers = []
        incoming = 1
        for channels in (32, 64, 128, 256, 512):
            layers += [nn.Conv2d(incoming, channels, 3, stride=2, padding=1),
                       nn.GroupNorm(8, channels), nn.GELU(),
                       nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                       nn.Conv2d(channels, channels, 1), nn.GELU()]
            incoming = channels
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1))
        self.projection = nn.Linear(512, 256)
        self.field_identity = nn.Parameter(torch.randn(1, fields, 256) * .02)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(256, 8, 1024, dropout=0., activation='gelu',
                                       batch_first=True, norm_first=True)
            for _ in range(4)])
        self.raw_branch = (nn.Sequential(nn.Linear(147, 512), nn.GELU(), nn.Linear(512, 256))
                           if system == 'nikon' else None)
        self.head = nn.Sequential(nn.LayerNorm(fields*256), nn.Linear(fields*256, 512),
                                  nn.GELU(), nn.Linear(512, outputs))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image, raw=None):
        if self.system == 'nikon':
            prediction = self.base(image, raw)
            normalized = torch.asinh(image)
        else:
            prediction = self.base(image)
            normalized = torch.asinh((image-self.base.reference)/1024.)
        batch, fields, height, width = normalized.shape
        tokens = self.projection(self.encoder(normalized.reshape(batch*fields, 1, height, width)).flatten(1))
        tokens = tokens.reshape(batch, fields, 256) + self.field_identity
        if raw is not None:
            native = ((raw-self.base.raw_mean)/self.base.raw_std).flatten(1)
            tokens = tokens + self.raw_branch(native)[:, None]
        for layer in self.layers:
            tokens = layer(tokens)
        return prediction + self.head(tokens.flatten(1))


class System:
    def __init__(self, name):
        self.name = name
        if name == 'nikon':
            self.source = ROOT / 'nikon_coordinated_center_20261005'
            self.legacy = load_module('archived_nikon_job', self.source/'job.py')
            self.payload = torch.load(self.source/'candidate.pt', map_location='cpu', weights_only=False)
            self.optics = self.legacy.NikonTensorOptics(device='cuda')
            self.threshold = .07
        else:
            self.source = ROOT / 'trepan_restored_joint_20261008'
            sys.path.insert(0, str(self.source))
            from network import JointNetwork
            from fast_optics import FastOptics
            from geometry import Geometry
            self.network_class = JointNetwork
            self.payload = torch.load(self.source/'training/candidate.pt', map_location='cpu', weights_only=False)
            self.optics = FastOptics(grid=65, device='cuda')
            self.optics32 = FastOptics(grid=65, device='cuda').float()
            self.geometry = Geometry()
            self.threshold = .08

    def model(self, variant):
        if self.name == 'nikon':
            base = self.legacy.CoordinatedNikon(self.legacy.previous.payloads('ordinary'),
                                               self.payload['raw_mean'], self.payload['raw_std'])
            base.load_state_dict(self.payload['state_dict'])
        else:
            base = self.network_class(self.payload['reference'])
            base.load_state_dict(self.payload['model'])
        return (base if variant == 'standard' else LargerNetwork(base, self.name)).cuda()

    def load_data(self, pilot=False):
        if self.name == 'nikon':
            sources = self.legacy.sources()
            if pilot:
                sources = [sources[0], sources[1], sources[40]]
            def read(item):
                path, group, training, old = item
                with np.load(path) as z:
                    assert str(z['sampling_mode']) == 'joint_context'
                    return (z['maps'].astype(np.float32), z['raw'].astype(np.float32),
                            z['targets'].astype(np.float64), z['seeds'].copy(),
                            np.full(len(z['seeds']), training), np.full(len(z['seeds']), group))
            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(read, sources))
            image, raw, q, ids, train, cohort = [np.concatenate([row[i] for row in rows]) for i in range(6)]
            assert np.isfinite(image).all() and np.isfinite(raw).all() and np.isfinite(q).all()
            assert len(np.unique(ids)) == len(ids)
            if not pilot:
                assert train.sum() == 24000 and (~train).sum() == 6000
                split = json.loads((self.legacy.PREVIOUS/'nikon/split_manifest.json').read_text(encoding='utf-8'))
                assert np.array_equal(ids[~train], split['validation_seeds'])
            files = [str(item[0]) for item in sources]
        else:
            files = sorted((self.source/'data').glob('cohort*.npz'))
            if pilot:
                files = [self.source/'data/cohort0_000.npz', self.source/'data/cohort0_140.npz']
            # Allocate once: avoid doubling the 4.4 GB image dataset at concatenate.
            image = np.empty((len(files)*50, 9, 128, 128), dtype=np.uint16)
            rows = []
            for i, path in enumerate(files):
                with np.load(path) as z:
                    assert z['speckles'].shape == (50, 9, 128, 128)
                    image[i*50:(i+1)*50] = z['speckles']
                    rows.append((z['q'].copy(), z['indices'].copy(), z['cohort'].copy()))
            q, ids, cohort = [np.concatenate([row[i] for row in rows]) for i in range(3)]
            assert np.isfinite(q).all()
            train = ids < 7000
            assert len(np.unique(np.c_[cohort, ids], axis=0)) == len(ids)
            if not pilot:
                assert len(files) == 300 and train.sum() == 14000 and (~train).sum() == 1000
            raw = None
            files = [str(p) for p in files]
        return dict(image=image, raw=raw, q=q, ids=ids, train=train, cohort=cohort, files=files)

    def forward(self, model, data, indices):
        image = torch.as_tensor(data['image'][indices].astype(np.float32), device='cuda')
        target = torch.as_tensor(data['q'][indices], dtype=torch.float64, device='cuda')
        if self.name == 'nikon':
            raw = torch.as_tensor(data['raw'][indices], device='cuda')
            prediction = model(image, raw)
            base = model.base if isinstance(model, LargerNetwork) else model
            command = -(prediction*base.target_std+base.target_mean).clamp(-1.5, 1.5)
            pose = nn.functional.smooth_l1_loss(prediction, ((target-base.target_mean)/base.target_std).float(), beta=.25)
        else:
            prediction = model(image)
            command = -torch.tanh(prediction)
            pose = nn.functional.smooth_l1_loss(command, -target.float())
        return target+command.double(), target, command, pose

    def metrics(self, endpoint, initial):
        if self.name == 'nikon':
            m = self.optics.metrics(endpoint)
            absolute = torch.sqrt(m['absolute'].square().mean(-1).clamp_min(1e-20))
            difference = torch.sqrt(m['difference'].square().mean(-1).clamp_min(1e-20))
            fields = torch.cat((absolute[:, :1], difference[:, 1:]), 1)
            return fields, m['valid'] & torch.isfinite(fields).all(1), absolute
        fields, support = self.optics.metrics(endpoint)
        paths = torch.as_tensor([self.geometry.path(b, a) for a, b in
                                zip(initial.detach().cpu().numpy(), endpoint.detach().cpu().numpy())], device='cuda')
        valid = paths & (support.amin(1) >= .98) & torch.isfinite(fields).all(1)
        return fields, valid, fields

    def loss(self, model, data, indices):
        endpoint, initial, command, pose = self.forward(model, data, indices)
        if self.name == 'trepan':
            return self.trepan_fp32_loss(endpoint, initial, command, pose)
        # Keep invalid traces out of differentiable optics, retaining a penalty
        # and pose supervision for every sample (no failed-row filtering).
        with torch.no_grad():
            _, valid, _ = self.metrics(endpoint, initial)
        if bool(valid.any()):
            fields, _, _ = self.metrics(endpoint[valid], initial[valid])
            from nikon_objective import physical_per_sample
            physical = physical_per_sample(fields,self.threshold).sum()
            physical = (physical+100*(~valid).sum())/len(indices)
        else:
            physical = command.sum()*0+100.
        mechanical = self.optics.mechanical_loss(endpoint).mean() if self.name == 'trepan' else command.sum()*0
        result = physical+(.05 if self.name == 'trepan' else .02)*pose+10*mechanical
        return result, dict(physical=float(physical.detach()), pose=float(pose.detach()),
                            mechanical=float(mechanical.detach()), invalid=int((~valid).sum()))

    def trepan_fp32_loss(self, endpoint, initial, command, pose):
        from mixed_optics import residual, wrms, corrected_fp32_metrics
        # Accurate optical values and validity checks carry no autograd graph.
        # The full9field differentiable optical calculation is FP32.
        with torch.no_grad():
            reference, mask = residual(self.optics, endpoint)
            w64 = wrms(reference, mask)
            paths = torch.as_tensor([self.geometry.path(b, a) for a, b in
                zip(initial.cpu().numpy(), endpoint.detach().cpu().numpy())], device=endpoint.device)
            valid = paths & (mask.double().mean(-1).amin(1)>=.98) & torch.isfinite(w64).all(1)
        if bool(valid.any()):
            w = corrected_fp32_metrics(self.optics32, endpoint[valid], reference[valid], mask[valid])
            ell = torch.log1p((w/self.threshold).square())
            terms = ell/(1+ell)
            physical = ((.5*terms.mean(1)+.5*terms.amax(1)).sum()+2*(~valid).sum())/len(endpoint)
        else:
            physical = command.sum()*0+2.
        mechanical = self.optics32.mechanical_loss(endpoint.float()).mean()
        loss = physical+.05*pose+10*mechanical
        return loss, dict(physical=float(physical.detach()), pose=float(pose.detach()),
                          mechanical=float(mechanical.detach()), invalid=int((~valid).sum()))


def aggregate(fields, valid, absolute, threshold):
    good = valid & np.isfinite(fields).all(1)
    passing = good & (fields.max(1) < threshold)
    finite = fields[good]
    return dict(count=len(fields), invalid=int((~good).sum()),
                full_field_passes=int(passing.sum()), full_field_rate=float(passing.mean()),
                all_fields_below07_rate=float((good & (fields.max(1)<.07)).mean()),
                all_fields_below08_rate=float((good & (fields.max(1)<.08)).mean()),
                center07_rate=float((good & (fields[:, 0]<.07)).mean()),
                center_and_mean08_rate=float((good & (fields[:, 0]<.08) & (fields.mean(1)<.08)).mean()),
                field_mean=float(finite.mean()) if len(finite) else None,
                worst_field_mean=float(finite.max(1).mean()) if len(finite) else None,
                worst_field_p90=float(np.quantile(finite.max(1), .9)) if len(finite) else None,
                per_field_mean=finite.mean(0).tolist() if len(finite) else None,
                per_field_pass_rate=((fields<threshold) & good[:, None]).mean(0).tolist(),
                absolute_per_field_mean=absolute[good].mean(0).tolist() if good.any() else None,
                threshold_waves=threshold)


@torch.no_grad()
def evaluate(system, model, data, indices, batch, path=None):
    model.eval()
    fields, valids, absolutes, commands = [], [], [], []
    for start in range(0, len(indices), batch):
        endpoint, initial, command, _ = system.forward(model, data, indices[start:start+batch])
        w, valid, absolute = system.metrics(endpoint, initial)
        fields.append(w.cpu().numpy()); valids.append(valid.cpu().numpy())
        absolutes.append(absolute.cpu().numpy()); commands.append(command.cpu().numpy())
    w, valid, absolute, command = [np.concatenate(part) for part in (fields, valids, absolutes, commands)]
    result = aggregate(w, valid, absolute, system.threshold)
    result['cohorts'] = {str(group): aggregate(w[data['cohort'][indices]==group], valid[data['cohort'][indices]==group],
                         absolute[data['cohort'][indices]==group], system.threshold)
                         for group in np.unique(data['cohort'][indices])}
    if path:
        np.savez_compressed(path, indices=indices, original_ids=data['ids'][indices], cohort=data['cohort'][indices],
                            initial=data['q'][indices], command=command, fields=w, valid=valid, absolute=absolute)
    return result


def selection(result):
    return (-result['full_field_passes'], result['worst_field_p90'] or 1e9, result['field_mean'] or 1e9)


def make_optimizer(model):
    if isinstance(model, LargerNetwork):
        base_ids = {id(p) for p in model.base.parameters()}
        groups = [dict(params=list(model.base.parameters()), lr=1e-5, initial_lr=1e-5),
                  dict(params=[p for p in model.parameters() if id(p) not in base_ids], lr=1e-4, initial_lr=1e-4)]
    else:
        groups = [dict(params=list(model.parameters()), lr=1e-5, initial_lr=1e-5)]
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def curves(out, history):
    atomic(out/'history.json', history)
    with (out/'curves.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'loss', 'full_field_rate', 'center07_rate', 'field_mean', 'worst_field_p90', 'active_seconds'])
        for row in history:
            v = row['validation']
            writer.writerow([row['epoch'], row.get('loss'), v['full_field_rate'], v['center07_rate'], v['field_mean'],
                             v['worst_field_p90'], row['active_seconds']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r['epoch'] for r in history]
    for key in ('full_field_rate', 'center07_rate'):
        axes[0].plot(epochs, [100*r['validation'][key] for r in history], label=key)
    axes[0].set_ylabel('Validation recovery (%)'); axes[0].set_ylim(0, 101)
    for key in ('field_mean', 'worst_field_p90'):
        axes[1].plot(epochs, [r['validation'][key] for r in history], label=key)
    axes[1].set_ylabel('WRMS (waves)')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.legend(); ax.grid(alpha=.25)
    fig.suptitle(out.name+' / same archived selection validation set')
    fig.tight_layout(); fig.savefig(out/'curves.png', dpi=150); plt.close(fig)


def preflight(system, args):
    data = system.load_data(pilot=True)
    train = np.flatnonzero(data['train'])
    val = np.flatnonzero(~data['train'])
    results = []
    # Data and grid are unchanged in these discarded forward/backward probes.
    for variant in ('standard', 'large'):
        torch.manual_seed(args.seed)
        model = system.model(variant)
        model.eval()
        base = model.base if variant == 'large' else model
        if variant == 'large':
            with torch.no_grad():
                a = system.forward(model, data, train[:2])[2]
                b = system.forward(base, data, train[:2])[2]
                assert torch.equal(a, b), 'Added branch must preserve epoch-zero commands'
        optimizer = make_optimizer(model)
        torch.cuda.reset_peak_memory_stats()
        timings, losses = [], []
        for step in range(3):
            tick = active_time(); model.train(); optimizer.zero_grad(set_to_none=True)
            loss, parts = system.loss(model, data, train[:args.microbatch])
            assert torch.isfinite(loss), parts
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            timings.append(active_time()-tick); losses.append(float(loss.detach()))
        if variant == 'large':
            assert sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)>0
        tick = active_time()
        score = evaluate(system, model, data, val[:16], args.microbatch)
        eval_per_case = (active_time()-tick)/16
        result = dict(variant=variant, parameters=sum(p.numel() for p in model.parameters()),
                      microbatch=args.microbatch, step_seconds=timings, losses=losses,
                      eval_seconds_per_case=eval_per_case, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                      finite_gradients=True, discarded_probe_updates=3, probe_score=score)
        results.append(result)
        print(json.dumps(result), flush=True)
        del model, optimizer, base, loss
        torch.cuda.empty_cache()
    with torch.no_grad():
        zero = torch.zeros((1, 35 if system.name == 'nikon' else 40), dtype=torch.float64, device='cuda')
        nominal, valid, absolute = system.metrics(zero, zero)
    atomic(HERE/f'preflight_{system.name}.json', dict(status='passed', updated=stamp(), gpu=torch.cuda.get_device_name(),
            nominal_fields=nominal.cpu().tolist(), nominal_absolute=absolute.cpu().tolist(),
            nominal_valid=valid.cpu().tolist(), variants=results))


def train(system, args, out=None):
    out = Path(out) if out is not None else HERE/f'{system.name}_{args.variant}'
    out.mkdir(exist_ok=True)
    if (out/'summary.json').exists():
        print('Already complete: '+str(out), flush=True)
        return
    def state(status, **kw):
        try:
            atomic(out/'state.json', dict(status=status, pid=os.getpid(), updated=stamp(), **kw))
        except PermissionError as error:
            # Telemetry must not discard a valid optimizer update. Checkpoint
            # failures still propagate; next progress publication retries.
            print('STATE_WRITE_DEFERRED '+repr(error), flush=True)
    state('loading_archived_data')
    data = system.load_data()
    train_ids, val_ids = np.flatnonzero(data['train']), np.flatnonzero(~data['train'])
    atomic(out/'data_manifest.json', dict(files=data['files'], train_count=len(train_ids), validation_count=len(val_ids),
                                        split='Original archived split, unchanged', no_new_observations=True,
                                        validation_used_for_selection=True))
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    model = system.model(args.variant)
    optimizer = make_optimizer(model)
    config = vars(args).copy()
    config.update(parameters=sum(p.numel() for p in model.parameters()), fields=3 if system.name=='nikon' else 9,
                  loss=('0.5 mean + 0.5 worst pseudo-Huber center ABS / edge DIFF penalty; pose0.02, invalid100'
                        if system.name=='nikon' else '0.5 mean + 0.5 worst bounded log full-field penalty; fixed pose/mechanical terms'),
                  objective_version='nikon_fullfield_pseudohuber_v2' if system.name=='nikon' else 'trepan_bounded_log_v1',
                  selection='Most valid full-field passes; tie lowest worst-field P90 then field mean',
                  invalids_count_as_failures=True, warm_start_epoch=int(system.payload['epoch']),
                  checkpoint_source=str(system.source), gpu=torch.cuda.get_device_name(),
                  training_grid='original Nikon audit nodes / Trepan pupil grid65',
                  optics_implementation='fp32_optical_derivatives_detached_fp64_reference_v1' if system.name=='trepan' else 'archived_nikon_fp64',
                  model_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
    config.update(getattr(system,'trial_config',{}))
    epoch, next_batch, accumulated, processed, elapsed = 1, 0, 0., 0, 0.
    history, best = [], None
    if (out/'resume.pt').exists():
        saved = torch.load(out/'resume.pt', map_location='cpu', weights_only=False)
        if system.name=='nikon' and saved['config'].get('objective_version')!=config['objective_version']:
            raise RuntimeError('Nikon objective changed: keep the historical run; use a separately named run after explicit training authorization.')
        assert all(saved['config'][k]==config[k] for k in ('system', 'variant', 'epochs', 'batch', 'microbatch', 'seed'))
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        epoch, next_batch, accumulated, processed, elapsed = [saved[k] for k in ('epoch', 'next_batch', 'accumulated', 'processed', 'elapsed')]
        history, best = saved['history'], saved['best']
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    atomic(out/'config.json', config)
    def payload():
        return dict(model=model.state_dict(), config=config, system=system.name, variant=args.variant)
    def checkpoint():
        save_pt(out/'resume.pt', payload()|dict(optimizer=optimizer.state_dict(), epoch=epoch, next_batch=next_batch,
                    accumulated=accumulated, processed=processed, elapsed=elapsed, history=history, best=best,
                    rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
    if not history:
        state('evaluating_archived_center_model')
        tick = active_time()
        baseline = evaluate(system, model, data, val_ids, args.microbatch, out/'baseline_validation.npz')
        history = [dict(epoch=0, validation=baseline, active_seconds=active_time()-tick)]
        best = selection(baseline)
        save_pt(out/'candidate.pt', payload()|dict(epoch=0, validation=baseline))
        atomic(out/'baseline.json', baseline); curves(out, history); checkpoint()
    total_batches = math.ceil(len(train_ids)/args.batch)
    while epoch <= args.epochs:
        order = np.random.default_rng(args.seed+epoch).permutation(train_ids)
        factor = .1+.9*.5*(1+math.cos(math.pi*(epoch-1)/args.epochs))
        for group in optimizer.param_groups:
            group['lr'] = group['initial_lr']*factor
        for batch in range(next_batch, total_batches):
            tick = active_time(); model.train(); optimizer.zero_grad(set_to_none=True)
            indices = order[batch*args.batch:(batch+1)*args.batch]
            batch_loss = 0.
            for start in range(0, len(indices), args.microbatch):
                ix = indices[start:start+args.microbatch]
                loss, parts = system.loss(model, data, ix)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss: '+repr(parts))
                (loss*(len(ix)/len(indices))).backward()
                batch_loss += float(loss.detach())*len(ix)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step(); torch.cuda.synchronize()
            elapsed += active_time()-tick
            accumulated += batch_loss; processed += len(indices); next_batch = batch+1
            if next_batch%10==0 or next_batch==total_batches:
                checkpoint()
                state('training', epoch=epoch, epochs=args.epochs, batch=next_batch, batches=total_batches,
                      loss=accumulated/processed, epoch_active_seconds=elapsed,
                      remaining_training_seconds=elapsed/max(next_batch, 1)*(total_batches-next_batch+(args.epochs-epoch)*total_batches))
            if next_batch%10==0:
                print(json.dumps(dict(epoch=epoch, batch=next_batch, batches=total_batches, loss=accumulated/processed,
                                      active_seconds=elapsed)), flush=True)
        state('validation', epoch=epoch, epochs=args.epochs)
        tick = active_time()
        validation = evaluate(system, model, data, val_ids, args.microbatch, out/f'validation_{epoch:03d}.npz')
        elapsed += active_time()-tick
        if selection(validation) < tuple(best):
            best = selection(validation)
            save_pt(out/'candidate.pt', payload()|dict(epoch=epoch, validation=validation))
            atomic(out/'selected.json', dict(epoch=epoch, validation=validation))
        history.append(dict(epoch=epoch, loss=accumulated/processed, validation=validation, active_seconds=elapsed,
                            optics_implementation=config['optics_implementation']))
        curves(out, history)
        print(json.dumps(history[-1]), flush=True)
        save_pt(out/'last.pt', payload()|dict(epoch=epoch, validation=validation))
        epoch += 1; next_batch = 0; accumulated = 0.; processed = 0; elapsed = 0.
        checkpoint()
    selected = torch.load(out/'candidate.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model'])
    final = evaluate(system, model, data, val_ids, args.microbatch, out/'selected_validation.npz')
    atomic(out/'summary.json', dict(status='completed', assistant_review='pending', selected_epoch=selected['epoch'],
            validation=final, baseline=history[0]['validation'], parameters=config['parameters'],
            epochs=args.epochs, active_seconds=sum(h['active_seconds'] for h in history),
            validation_is_selection_set=True, independent_test=False, updated=stamp()))
    state('completed', selected_epoch=selected['epoch'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['preflight', 'train'])
    parser.add_argument('--system', choices=['nikon', 'trepan'], required=True)
    parser.add_argument('--variant', choices=['standard', 'large'], default='large')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--microbatch', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261024)
    args = parser.parse_args()
    if (HERE/'TRAINING_STOPPED.json').exists():
        raise SystemExit('Training and optimizer probes stopped by user; use the separate zero-training debug scripts.')
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        system = System(args.system)
        (preflight if args.stage=='preflight' else train)(system, args)
    except Exception:
        atomic(HERE/f'failure_{args.system}_{args.variant}_{args.stage}.json', dict(error=traceback.format_exc(), updated=stamp()))
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
