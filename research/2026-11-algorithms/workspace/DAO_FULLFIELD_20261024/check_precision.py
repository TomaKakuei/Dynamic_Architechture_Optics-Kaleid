"""Small actual-value/gradient probe; pause the existing GPU worker briefly."""
import json
from pathlib import Path
import time
import copy
import numpy as np
import psutil
import torch
from experiment import System, HERE, make_optimizer
from durable_io import atomic_json
from profile_resources import original_loss


def elapsed(fn):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter()-tick


def penalty(w):
    ell = torch.log1p((w/.08).square()); terms = ell/(1+ell)
    return (.5*terms.mean(1)+.5*terms.amax(1)).mean()


def probe():
    system = System('trepan')
    from fast_optics import FastOptics
    from mixed_optics import residual, wrms, corrected_fp32_metrics
    optical64 = system.optics
    optical32 = FastOptics(grid=65, device='cuda').float()
    data = system.load_data(pilot=True)
    model = system.model('large')
    saved = torch.load(HERE/'trepan_large/resume.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model']); model.eval()
    ids = np.arange(16)
    with torch.no_grad():
        endpoint, initial, _, _ = system.forward(model, data, ids)
    results = []
    for label, q0 in [('initial', initial), ('checkpoint_endpoint', endpoint)]:
        q = q0.detach().clone().requires_grad_()
        w64, support64 = optical64.metrics(q)
        l64 = penalty(w64)
        g64, = torch.autograd.grad(l64, q)
        with torch.no_grad():
            reference, mask = residual(optical64, q0)
        for mode in ('all_fp32', 'fp32_derivatives_reference_values'):
            q = q0.detach().clone().requires_grad_()
            if mode=='all_fp32':
                w, support = optical32.metrics(q.float())
            else:
                w = corrected_fp32_metrics(optical32, q, reference, mask)
                support = mask.double().mean(-1)
            loss = penalty(w)
            gradient, = torch.autograd.grad(loss, q)
            cosine = torch.nn.functional.cosine_similarity(g64.flatten(), gradient.flatten(), dim=0)
            row = dict(case=label, mode=mode, loss=float(loss), reference_loss=float(l64),
                loss_relative_error=float(abs(loss-l64)/l64.abs().clamp_min(1e-20)),
                max_wrms_error=float((w-w64).abs().max()),
                gradient_relative_l2=float(torch.linalg.vector_norm(gradient-g64)/torch.linalg.vector_norm(g64).clamp_min(1e-20)),
                gradient_cosine=float(cosine), support_equal=torch.equal(support, support64),
                all_field_pass_disagreements=int(((w.amax(1)<.08)!=(w64.amax(1)<.08)).sum()),
                finite_gradient=bool(torch.isfinite(gradient).all()))
            results.append(row)
    def mixed_loss():
        endpoint, initial, command, pose = system.forward(model, data, ids)
        with torch.no_grad():
            reference, mask = residual(optical64, endpoint)
            w64 = wrms(reference, mask)
            paths = torch.tensor([system.geometry.path(b, a) for a, b in zip(initial.cpu().numpy(), endpoint.detach().cpu().numpy())], device='cuda')
            valid = (mask.double().mean(-1).amin(1)>=.98) & torch.isfinite(w64).all(1) & paths
        assert valid.all()
        w = corrected_fp32_metrics(optical32, endpoint, reference, mask)
        physical = penalty(w)
        mechanical = optical32.mechanical_loss(endpoint.float()).mean()
        return physical+.05*pose+10*mechanical
    timings = {}
    for mode in ('fast_fp64', 'fp32_derivatives_reference_values'):
        model.load_state_dict(saved['model']); model.train()
        optimizer = make_optimizer(model)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = original_loss(system, model, data, ids) if mode=='fast_fp64' else mixed_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            return float(loss.detach())
        rows = []
        for repeat in range(5):
            value, seconds = elapsed(step)
            if repeat:
                rows.append(dict(seconds=seconds, loss=value))
        timings[mode] = rows
        del optimizer
        torch.cuda.empty_cache()
    result = dict(checks=results, timings=timings, model_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
                  checkpoint_updates=0, actual_fp32_probe=True)
    atomic_json(HERE/'performance_20261024/precision_check.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    queue = json.loads((HERE/'state.json').read_text(encoding='utf-8'))
    worker = psutil.Process(queue['child_pid'])
    assert 'experiment.py' in ' '.join(worker.cmdline()) and queue['job']=='nikon_large'
    before = json.loads((HERE/'nikon_large/state.json').read_text(encoding='utf-8'))
    worker.suspend()
    tick = time.time()
    try:
        time.sleep(.3)
        probe()
    finally:
        worker.resume()
        atomic_json(HERE/'performance_20261024/nikon_timing_interruption.json', dict(
            pid=worker.pid, state_before=before, suspended_seconds=time.time()-tick,
            excluded_epoch_from_speed_estimates=before.get('epoch'),
            reason='Serial GPU precision diagnostics; raw training timing retained'))


if __name__=='__main__':
    main()
