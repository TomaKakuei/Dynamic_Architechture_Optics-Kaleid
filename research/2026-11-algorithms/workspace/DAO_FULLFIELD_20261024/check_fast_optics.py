"""Bounded full-field value/gradient comparison and serial throughput probe."""
import copy
import json
import time
import numpy as np
import torch
from experiment import System, HERE, make_optimizer
from durable_io import atomic_json
from profile_resources import original_loss


def measure(fn):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return value, time.perf_counter()-tick


def main():
    system = System('trepan')
    from fast_optics import FastOptics
    from optics import Optics
    fast = FastOptics(grid=65, device='cuda')
    native = Optics(grid=65, device='cuda')
    data = system.load_data(pilot=True)
    model = system.model('large')
    saved = torch.load(HERE/'trepan_large/resume.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model']); model.eval()
    ids = np.arange(16)
    with torch.no_grad():
        endpoint, initial, _, _ = system.forward(model, data, ids)
    checks = []
    for label, q0 in [('initial', initial), ('checkpoint_endpoint', endpoint)]:
        outputs = []
        for optical in (native, fast):
            q = q0.detach().clone().requires_grad_()
            w, support = optical.metrics(q)
            ell = torch.log1p((w/.08).square()); terms = ell/(1+ell)
            loss = (.5*terms.mean(1)+.5*terms.amax(1)).mean()
            gradient, = torch.autograd.grad(loss, q)
            outputs.append((w.detach(), support.detach(), gradient.detach(), float(loss.detach())))
        old, new = outputs
        row = dict(case=label, max_wrms_difference=float((old[0]-new[0]).abs().max()),
                   support_equal=torch.equal(old[1], new[1]), loss_difference=abs(old[3]-new[3]),
                   gradient_relative_l2=float(torch.linalg.vector_norm(old[2]-new[2])/torch.linalg.vector_norm(old[2]).clamp_min(1e-20)),
                   max_gradient_difference=float((old[2]-new[2]).abs().max()))
        assert row['support_equal'] and row['max_wrms_difference']<1e-7 and row['gradient_relative_l2']<1e-5, row
        checks.append(row)
    timings = {}
    model_gradient_check = []
    # Same checkpoint, optimizer state and input for both measured paths.
    for label, optical in [('native', native), ('explicit_rotation', fast)]:
        system.optics = optical
        model.load_state_dict(saved['model']); model.train()
        optimizer = make_optimizer(model)
        optimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        def step():
            optimizer.zero_grad(set_to_none=True)
            loss = original_loss(system, model, data, ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            return float(loss.detach())
        values = []
        for repeat in range(5):
            loss, seconds = measure(step)
            if repeat:
                values.append(dict(seconds=seconds, loss=loss))
        timings[label] = values
        del optimizer
        torch.cuda.empty_cache()
    ratio = np.median([x['seconds'] for x in timings['native']])/np.median([x['seconds'] for x in timings['explicit_rotation']])
    result = dict(status='passed', checks=checks, timings=timings, speed_ratio=float(ratio),
                  microbatch=16, fields=9, optics_dtype='float64', checkpoint_updates=0)
    atomic_json(HERE/'performance_20261024/fast_optics_check.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__=='__main__':
    main()
