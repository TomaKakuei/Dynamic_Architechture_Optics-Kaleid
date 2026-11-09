"""CPU-only diagnosis from saved predictions; no training or network inference."""
from pathlib import Path
import os
import sys
import json
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
import numpy as np
import torch
import experiment as exp


def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    legacy = exp.load_module('paused_sensor_scope_check', exp.ROOT / 'nikon_coordinated_center_20261005/job.py')
    native = legacy.NikonTensorOptics(nodes=legacy.nikon35.NODES,
        fields=legacy.nikon35.base.corrected.CONTROL_FIELDS_MM, device='cpu')
    folder = BASE / 'nikon_large_debug_20261025/trajectory_4ep'
    epoch = json.loads((folder / 'selected.json').read_text())['epoch']
    with np.load(folder / f'validation_{epoch:03d}.npz') as z:
        positions = np.linspace(0, len(z['initial']) - 1, 32, dtype=int)
        initial, command, fields, valid, indices = [z[key][positions].copy()
            for key in ('initial', 'command', 'fields', 'valid', 'indices')]
        whole = dict(count=len(z['fields']),
            center_pass=int((z['valid'] & (z['fields'][:, 0] < .07)).sum()),
            edge_minus_pass=int((z['valid'] & (z['fields'][:, 1] < .07)).sum()),
            edge_plus_pass=int((z['valid'] & (z['fields'][:, 2] < .07)).sum()),
            full_field_pass=int((z['valid'] & (z['fields'].max(1) < .07)).sum()))
    sources = legacy.sources()
    raw = []
    for index in indices:
        with np.load(sources[int(index) // 100][0]) as z:
            raw.append(z['raw'][int(index) % 100])
    raw = np.asarray(raw)
    with torch.no_grad():
        q = torch.tensor(initial, dtype=torch.float64)
        p = torch.tensor(-command, dtype=torch.float64)
        truth, estimated, endpoint = [native.metrics(x) for x in (q, p, q-p)]
        native_fields = torch.cat((endpoint['absolute'][:, :1].square().mean(-1).sqrt(),
            endpoint['difference'][:, 1:].square().mean(-1).sqrt()), 1).numpy()
        true_signal = truth['difference'].numpy()
        estimate_signal = estimated['difference'].numpy()
        native_valid = endpoint['valid'].numpy()
    probe = dict(count=32,
        selection='32 equally spaced positions in unchanged 6000 validation list; no optimizer or network inference',
        positions=positions.tolist(), input_fields_mm=list(native.fields),
        report_fields_mm=list(legacy.NikonTensorOptics(device='cpu').fields),
        native_nodes=len(native.nodes), extra_center_abs_nodes=len(legacy.nikon35.AUDIT_NODES),
        true_pose_sensor_fit_rms_waves=float(np.sqrt(np.mean((true_signal-raw)**2))),
        estimated_pose_sensor_fit_rms_waves=float(np.sqrt(np.mean((estimate_signal-raw)**2))),
        native_per_field_mean_waves=native_fields.mean(0).tolist(),
        report_per_field_mean_waves=fields.mean(0).tolist(),
        native_fullfield_pass=int((native_valid & (native_fields.max(1) < .07)).sum()),
        report_fullfield_pass=int((valid & (fields.max(1) < .07)).sum()),
        native_edges_both_pass=int((native_valid & (native_fields[:, 1:].max(1) < .07)).sum()),
        report_edges_both_pass=int((valid & (fields[:, 1:].max(1) < .07)).sum()),
        rows=[dict(position=int(positions[i]), native_fields=native_fields[i].tolist(),
            report_fields=fields[i].tolist(),
            input_fit_rms=float(np.sqrt(np.mean((estimate_signal[i]-raw[i])**2)))) for i in range(32)])
    comparison = BASE / 'nikon_short_training/exact_local_response/original_protocol_comparison'
    with np.load(comparison / 'idea_selected.npz') as one, np.load(comparison / 'idea_selected_measured_loop.npz') as loop:
        assert np.array_equal(one['original_ids'], loop['original_ids'])
        assert np.array_equal(one['initial'], loop['initial'])
        paired = []
        for label, z in [('one_acquisition', one), ('measured_loop', loop)]:
            f, v = z['fields'], z['valid']
            paired.append(dict(route=label, count=len(f),
                center_pass=int((v & (f[:, 0] < .07)).sum()),
                fullfield_pass=int((v & (f.max(1) < .07)).sum()), per_field_mean=f.mean(0).tolist()))
    values = np.array([.0356, .07, .1, .2, 1.])
    ell = np.log1p((values/.07)**2)
    derivative = (2*values/(.07**2+values**2))/(1+ell)**2
    result = dict(status='completed_zero_training_review', epoch=epoch,
        latest6000_saved_scores=whole, matched_historical6000_comparison=paired,
        same_poses_and_ids_verified=True, probe=probe,
        active_loss='ell=log1p((field/.07)^2); term=ell/(1+ell); 0.5 mean+0.5 worst; pose 0.02',
        loss_term_derivative_wrt_field=[dict(field_waves=float(x), derivative=float(y))
            for x, y in zip(values, derivative)],
        optimizer_updates=0, network_inferences=0, new_optical_states=96, device='cpu',
        cuda_initialized=torch.cuda.is_initialized(), seconds=time.perf_counter()-start,
        training_remains_paused=(BASE/'nikon_large_debug_20261025/STOP').exists())
    assert not result['cuda_initialized'] and result['training_remains_paused']
    (HERE/'sensor_scope_and_loss.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    compact = {k: v for k, v in result.items() if k != 'probe'}
    compact['probe'] = {k: v for k, v in probe.items() if k not in ('rows', 'positions')}
    print(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
