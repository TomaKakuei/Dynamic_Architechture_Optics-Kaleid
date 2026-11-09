"""Short serial profile of current Trepan-large; no training checkpoint writes."""
from pathlib import Path
import json
import time
import torch
import numpy as np
import psutil
from experiment import System, make_optimizer, HERE, active_time
from durable_io import atomic_json

OUT = HERE/'performance_20261024'
OUT.mkdir(exist_ok=True)


def original_loss(system, model, data, indices):
    """Frozen pre-optimization loss for reproducible resource comparisons."""
    endpoint, initial, command, pose = system.forward(model, data, indices)
    with torch.no_grad():
        _, valid, _ = system.metrics(endpoint, initial)
    if bool(valid.any()):
        fields, _, _ = system.metrics(endpoint[valid], initial[valid])
        ell = torch.log1p((fields/system.threshold).square()); terms = ell/(1+ell)
        physical = ((.5*terms.mean(1)+.5*terms.amax(1)).sum()+2*(~valid).sum())/len(indices)
    else:
        physical = command.sum()*0+2.
    mechanical = system.optics.mechanical_loss(endpoint).mean()
    return physical+.05*pose+10*mechanical


def elapsed(fn):
    torch.cuda.synchronize()
    tick = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter()-tick


def main():
    torch.manual_seed(20261024)
    system = System('trepan')
    from optics import Optics
    system.optics = Optics(grid=65, device='cuda')
    data = system.load_data(pilot=True)
    ids = np.arange(16)
    model = system.model('large')
    saved = torch.load(HERE/'trepan_large/resume.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    optimizer = make_optimizer(model)
    optimizer.load_state_dict(saved['optimizer'])
    del saved
    process = psutil.Process()
    def step():
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = original_loss(system, model, data, ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach())
    for _ in range(2):
        elapsed(step)
    steps = []
    for _ in range(4):
        cpu0 = process.cpu_times()
        loss, seconds = elapsed(step)
        cpu1 = process.cpu_times()
        steps.append(dict(seconds=seconds, loss=loss, cpu_seconds=cpu1.user+cpu1.system-cpu0.user-cpu0.system))
    model.train(); optimizer.zero_grad(set_to_none=True)
    (endpoint, initial, command, pose), network_seconds = elapsed(lambda: system.forward(model, data, ids))
    def detached_optics():
        with torch.no_grad():
            return system.optics.metrics(endpoint)
    (initial_fields, support), nongrad_optics_seconds = elapsed(detached_optics)
    (initial_cpu, endpoint_cpu), transfer_seconds = elapsed(lambda: (initial.detach().cpu().numpy(), endpoint.detach().cpu().numpy()))
    tick = time.perf_counter()
    paths = [system.geometry.path(b, a) for a, b in zip(initial_cpu, endpoint_cpu)]
    geometry_seconds = time.perf_counter()-tick
    (fields, support), grad_optics_seconds = elapsed(lambda: system.optics.metrics(endpoint))
    ell = torch.log1p((fields/.08).square()); terms = ell/(1+ell)
    physical = (.5*terms.mean(1)+.5*terms.amax(1)).mean()
    mechanical, mechanical_seconds = elapsed(lambda: system.optics.mechanical_loss(endpoint).mean())
    loss = physical+.05*pose+10*mechanical
    _, backward_seconds = elapsed(lambda: loss.backward())
    def update():
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
    _, optimizer_seconds = elapsed(update)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profile:
        elapsed(step)
    averages = profile.key_averages()
    (OUT/'torch_profile_cpu.txt').write_text(averages.table(sort_by='self_cpu_time_total', row_limit=25), encoding='utf-8')
    (OUT/'torch_profile_gpu.txt').write_text(averages.table(sort_by='self_cuda_time_total', row_limit=25), encoding='utf-8')
    events = []
    for event in averages:
        events.append(dict(name=event.key, calls=event.count, cpu_total_us=event.cpu_time_total,
            self_cpu_us=event.self_cpu_time_total,
            gpu_total_us=getattr(event, 'device_time_total', getattr(event, 'cuda_time_total', 0)),
            self_gpu_us=getattr(event, 'self_device_time_total', getattr(event, 'self_cuda_time_total', 0))))
    events.sort(key=lambda event: event['self_gpu_us'], reverse=True)
    # Isolate original float64 ray tracing at the same16states for1vs9fields.
    center = type(system.optics)(grid=65, fields=[0], device='cuda')
    optical_timings = {}
    for label, optical in [('center_only', center), ('nine_fields', system.optics)]:
        values = []
        for repeat in range(4):
            q = endpoint.detach().clone().requires_grad_()
            def optics_step():
                w, support = optical.metrics(q)
                w.mean().backward()
            _, seconds = elapsed(optics_step)
            if repeat:
                values.append(seconds)
        optical_timings[label] = values
    result = dict(gpu=torch.cuda.get_device_name(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, microbatch=16, fields=9, optics_dtype='float64',
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        threads=torch.get_num_threads(), current_steps=steps,
        phases=dict(network_and_transfer=network_seconds, nongrad_optics=nongrad_optics_seconds,
                    state_copy_to_cpu=transfer_seconds, geometry_one_pass=geometry_seconds,
                    grad_optics_forward=grad_optics_seconds, mechanical=mechanical_seconds,
                    combined_backward=backward_seconds, optimizer=optimizer_seconds),
        original_code_geometry_checks_per_microbatch=2,
        optical_workload_comparison=optical_timings, profile_top_gpu=events[:30],
        paths_valid=all(paths), checkpoint_updates=0,
        note='Discarded diagnostic updates only. Phases synchronize separately; profiler adds overhead. No4050device tested.')
    atomic_json(OUT/'profile.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='profile_top_gpu'}, indent=2), flush=True)


if __name__=='__main__':
    main()
