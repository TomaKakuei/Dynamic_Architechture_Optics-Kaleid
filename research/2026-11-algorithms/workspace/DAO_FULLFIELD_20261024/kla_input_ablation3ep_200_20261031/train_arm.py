"""Three-epoch old-loss input ablation; automatically score the same fixed200."""
from pathlib import Path
import json
import os
import sys
import traceback
import msvcrt

ROOT = Path(__file__).resolve().parent
ARM = sys.argv[1]
OUT = ROOT / ARM
OUT.mkdir(exist_ok=True)
DAO = ROOT.parent
from losses import OpticalLoss, ARMS
assert ARM in ARMS
SOURCE = DAO.parent / 'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(DAO))
from common import np, stamp, clock, awake
from durable_io import atomic_json, save_torch, temporary, retry_io
from input_network import restore
from optics import Optics, torch
from geometry import Geometry

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
EPOCHS, BATCH, MICRO = 3, 32, 16


def state(stage, **kw):
    try:
        atomic_json(OUT / 'state.json', dict(stage=stage, pid=os.getpid(), updated=stamp(), **kw))
    except PermissionError as exc:
        print(f'Telemetry deferred: {exc!r}', flush=True)


def load_npz(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def save_npz(path, **arrays):
    tmp = temporary(path)
    with tmp.open('wb') as f:
        np.savez_compressed(f, **arrays)
    retry_io(lambda: os.replace(tmp, path))


def stopped():
    return (OUT / 'STOP_REQUESTED.json').exists() or (ROOT / 'STOP_REQUESTED.json').exists()


def summarize(a):
    good = a['valid'] & (a['score'] >= .45)
    return dict(count=len(good), passes=int(good.sum()), rate=float(good.mean()),
                invalid=int((~a['valid']).sum()), q_mean=float(a['score'].mean()),
                wrms_mean=float(a['wrms'].mean()),
                worst_field_wrms_mean=float(a['wrms'].max(1).mean()))


def paired(a, b):
    before, after = a['valid'] & (a['score'] >= .45), b['valid'] & (b['score'] >= .45)
    gained, lost = ~before & after, before & ~after
    usable = a['valid'] & b['valid']
    dw = b['wrms'].mean(1) - a['wrms'].mean(1)
    dq = b['score'] - a['score']
    return dict(gained=int(gained.sum()), lost=int(lost.sum()), net=int(after.sum() - before.sum()),
                gained_ids=b['indices'][gained].tolist(), lost_ids=b['indices'][lost].tolist(),
                valid_pairs=int(usable.sum()), wrms_improved=int((usable & (dw < 0)).sum()),
                q_improved=int((usable & (dq > 0)).sum()),
                wrms_improved_but_q_worse=int((usable & (dw < 0) & (dq < 0)).sum()),
                mean_wrms_change=float(dw[usable].mean()) if usable.any() else None)


def main():
    if (OUT / 'result.json').exists():
        print('Already complete; do not rerun.'); return
    if stopped():
        state('stopped_before_start'); return
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    plan = load_npz(ROOT / 'selection.npz')
    train, rows = plan['train'], plan['rows']
    meta = load_npz(SOURCE / 'cache/metadata.npz')
    old = load_npz(SOURCE / 'training/joint_validation3000.npz')
    assert np.array_equal(old['indices'][rows], plan['indices'])
    images = np.load(SOURCE / 'cache/images.npy', mmap_mode='r')
    geometry, optic = Geometry(), Optics(grid=65, device='cuda')
    objective = OpticalLoss(optic, ARM, out=OUT)
    torch.manual_seed(protocol['model_seed'])
    torch.cuda.manual_seed_all(protocol['model_seed'])
    model, payload = restore(protocol['initial_checkpoint'], arm=ARM)
    assert all(p.dtype == torch.float32 for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    batches = (len(train) + BATCH - 1) // BATCH
    history, first_epoch, first_batch = [], 1, 0
    active, accum, processed = 0., np.zeros(3), 0
    clipped, grad_sum, grad_max = 0, 0., 0.
    if (OUT / 'resume.pt').exists():
        r = torch.load(OUT / 'resume.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(r['model']); optimizer.load_state_dict(r['optimizer'])
        first_epoch, first_batch = r['next_epoch'], r['next_batch']
        history, active, accum, processed = r['history'], r['active_seconds'], np.array(r['accum']), r['processed']
        clipped, grad_sum, grad_max = r['clipped_updates'], r['grad_norm_sum'], r['grad_norm_max']
        torch.set_rng_state(r['torch_rng']); torch.cuda.set_rng_state_all(r['cuda_rng'])

    def package(epoch):
        return dict(model=model.state_dict(), reference=payload['reference'], scales=payload.get('scales'),
                    epoch=epoch, arm=ARM, input_ablation_arm=ARM, source_version='old_wfe_stable_epoch4', protocol=protocol)

    def checkpoint(ep, ba):
        save_torch(OUT / 'resume.pt', package(min(ep, EPOCHS)) | dict(
            optimizer=optimizer.state_dict(), next_epoch=ep, next_batch=ba, history=history,
            active_seconds=active, accum=accum.tolist(), processed=processed,
            clipped_updates=clipped, grad_norm_sum=grad_sum, grad_norm_max=grad_max,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))

    state('initializing', train_samples=len(train), test_samples=len(rows), total_updates=EPOCHS*batches)
    for epoch in range(first_epoch, EPOCHS + 1):
        order = np.random.default_rng(2026102900 + epoch).permutation(train)
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        for batch in range(first_batch, batches):
            if stopped():
                checkpoint(epoch, batch); state('paused_by_stop', epoch=epoch, batch=batch); return
            bt = clock()
            ids = order[batch*BATCH:(batch+1)*BATCH]
            optimizer.zero_grad(set_to_none=True)
            parts = np.zeros(3)
            for micro in range(0, len(ids), MICRO):
                ix = ids[micro:micro+MICRO]
                target = torch.tensor(meta['q'][ix], device='cuda', dtype=torch.float64)
                observed = torch.tensor(images[ix].astype(np.float32), device='cuda')
                cmd = model.command(observed)
                endpoint = target + cmd.double()
                phase, mask = optic.residual(endpoint)
                w = mask.double()
                wrms = ((phase.square()*w).sum(-1)/w.sum(-1).clamp_min(1)+1e-20).sqrt()
                # Optical loss and auxiliary coefficients are exactly the archived control.
                physical, loss_components = objective(phase, mask, wrms, cmd, target,
                    observed=observed, ids=ix, epoch=epoch)
                path = torch.tensor([geometry.path(b, a) for a, b in zip(
                    target.detach().cpu().numpy(), endpoint.detach().cpu().numpy())], device='cuda')
                valid = (w.mean(-1) >= .98).all(-1) & path & torch.isfinite(physical)
                physical = torch.where(valid, physical, 2 + torch.nan_to_num(physical, nan=1., posinf=1.))
                pose = torch.nn.functional.smooth_l1_loss(cmd, -target.float(), reduction='none').mean(1)
                mechanical = geometry.penalty(endpoint)
                loss = (physical + .05*pose + 10*mechanical).mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss; stopped before optimizer update')
                (loss*(len(ix)/len(ids))).backward()
                parts += len(ix)*np.array([float(loss.detach()), float(physical.detach().mean()), float(wrms.detach().mean())])
                del phase, mask, w, wrms, loss, physical, cmd, endpoint, observed
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
            clipped += int(gn > 5.); grad_sum += gn; grad_max = max(grad_max, gn)
            optimizer.step(); torch.cuda.synchronize()
            active += clock()-bt; accum += parts; processed += len(ids)
            updates = (epoch-1)*batches + batch+1
            if (batch+1) % 20 == 0 or batch+1 == batches:
                checkpoint(epoch, batch+1)
            if (batch+1) % 10 == 0 or batch == 0 or batch+1 == batches:
                rate = active/updates
                info = dict(epoch=epoch, epochs=EPOCHS, batch=batch+1, batches=batches,
                            updates=updates, total_updates=EPOCHS*batches, active_train_seconds=active,
                            seconds_per_update=rate, remaining_train_seconds=(EPOCHS*batches-updates)*rate,
                            train_means=(accum/processed).tolist(), clipped_updates=clipped,
                            arm=ARM, last_microbatch_components=loss_components, network_dtype='float32', gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30)
                if objective.image is not None:
                    info['image_loss'] = objective.image.diagnostics()
                state('training', **info); print(json.dumps(info), flush=True)
        history.append(dict(epoch=epoch, train_means=(accum/processed).tolist(), active_train_seconds=active))
        atomic_json(OUT / 'history.json', history)
        save_torch(OUT / 'last.pt', package(epoch))
        first_batch, accum, processed = 0, np.zeros(3), 0
        checkpoint(epoch+1, 0)
    save_torch(OUT / 'candidate.pt', package(EPOCHS))
    atomic_json(OUT / 'training_complete.json', dict(epochs=EPOCHS, updates=EPOCHS*batches,
        active_train_seconds=active, clipped_updates=clipped, mean_grad_norm=grad_sum/(EPOCHS*batches),
        max_grad_norm=grad_max, checkpoint='candidate.pt', history=history))
    if objective.image is not None:
        atomic_json(OUT / 'image_loss_timing.json', objective.image.diagnostics())
    del optimizer, objective
    model.eval()
    eval_start = clock()

    @torch.no_grad()
    def metrics(op, command, size):
        pieces = []
        for first in range(0, len(rows), size):
            ix = rows[first:first+size]
            ep = old['initial'][ix] + command[first:first+size].astype(np.float64)
            m = op.metrics(torch.tensor(ep, device='cuda', dtype=torch.float64))
            a = {k: v.cpu().numpy() for k, v in m.items()}
            a['valid'] &= np.array([geometry.path(b, a0) for a0, b in zip(old['initial'][ix], ep)])
            pieces.append(a)
        return {k: np.concatenate([a[k] for a in pieces]) for k in pieces[0]}

    dense129, dense257 = Optics(grid=129, device='cuda'), Optics(grid=257, device='cuda')

    @torch.no_grad()
    def evaluate(name, command, coarse):
        target = (ROOT if name == 'bounded_control' else OUT) / f'{name}200.npz'
        if target.exists():
            return load_npz(target)
        state('evaluating200', arm=name, grid=129)
        a = metrics(dense129, command, 4)
        border = np.where((abs(a['score']-.45) < .02) | ((a['score'] >= .45) != (coarse['score'] >= .45)))[0]
        a['score129'] = a['score'].copy()
        a['valid'] &= coarse['valid']
        for first in range(0, len(border), 2):
            ix = border[first:first+2]
            ep = old['initial'][rows[ix]] + command[ix].astype(np.float64)
            m = dense257.metrics(torch.tensor(ep, device='cuda', dtype=torch.float64))
            a['score'][ix] = m['score'].cpu().numpy()
            a['valid'][ix] &= m['valid'].cpu().numpy()
        a.update(command=command, rows=rows, indices=plan['indices'], border=border,
                 score65=coarse['score'], valid65=coarse['valid'])
        save_npz(target, **a)
        print(json.dumps(dict(arm=name, metric=summarize(a))), flush=True)
        return a

    command_file = OUT / 'candidate_commands200.npz'
    if command_file.exists():
        command = load_npz(command_file)['command']
    else:
        with torch.no_grad():
            command = np.concatenate([model.command(torch.tensor(images[plan['indices'][first:first+16]].astype(np.float32),
                                    device='cuda')).cpu().numpy() for first in range(0, len(rows), 16)])
        save_npz(command_file, command=command, indices=plan['indices'], rows=rows)
    del model
    torch.cuda.empty_cache()
    candidate = evaluate('candidate', command, metrics(optic, command, 16))
    saved_control = load_npz(ROOT / 'control200_grid65.npz')
    control = evaluate('bounded_control', saved_control['command'], saved_control)
    baseline = load_npz(ROOT / 'baseline200.npz')
    result = dict(status='complete', assistant_interpretation='pending', protocol_file='protocol.json',
        baseline=summarize(baseline), bounded_control=summarize(control), candidate=summarize(candidate), arm=ARM,
        direct_vs_control=paired(control, candidate), direct_vs_initial=paired(baseline, candidate),
        control_vs_initial=paired(baseline, control), training_active_seconds=active,
        evaluation_active_seconds=clock()-eval_start, inference='one packet, one prediction, no loop',
        automatic_deployment=False, automatic_full3000=False,
        limitations=['Reused selection validation; not a fresh independent test',
                     'One seed and three epochs; masked inputs warm-start the full-field model',
                     'Same-packet feature changes cannot establish physical measurement sufficiency'])
    atomic_json(OUT / 'result.json', result)
    text = 'KLA input three-epoch ablation; fixed paired200. Generated results, assistant interpretation pending.\n'
    for name in ('baseline', 'bounded_control', 'candidate'):
        a = result[name]
        text += f"{name}: {a['passes']}/200 = {100*a['rate']:.2f}%; mean WRMS {a['wrms_mean']:.8f} waves; invalid {a['invalid']}.\n"
    text += '\n' + json.dumps(result, indent=2)
    retry_io(lambda: (OUT / 'GENERATED_REPORT.txt').write_text(text, encoding='utf-8'))
    state('complete', result_file='result.json', assistant_interpretation='pending',
          direct_passes=result['candidate']['passes'], control_passes=result['bounded_control']['passes'])
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    with (OUT / 'run.lock').open('a+b') as lock:
        lock.seek(0)
        if lock.read(1) == b'':
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            awake(); main()
        except Exception as exc:
            state('failed', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            awake(False)
