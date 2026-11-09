"""Isolate field position using identical 49-node pupil sampling."""
from audit_sensor_scope import *


def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    legacy = exp.load_module('same_nodes_outer_fields', exp.ROOT/'nikon_coordinated_center_20261005/job.py')
    optics = legacy.NikonTensorOptics(nodes=legacy.nikon35.NODES, fields=(0., -.1, .1), device='cpu')
    folder = BASE/'nikon_large_debug_20261025/trajectory_4ep'
    epoch = json.loads((folder/'selected.json').read_text())['epoch']
    with np.load(folder/f'validation_{epoch:03d}.npz') as z:
        q, command, reported, good, ids = [z[k].copy() for k in
            ('initial', 'command', 'fields', 'valid', 'original_ids')]
    edges, validity = [], []
    with torch.no_grad():
        for offset in range(0, len(q), 128):
            endpoint = torch.tensor(q[offset:offset+128], dtype=torch.float64)
            endpoint += torch.tensor(command[offset:offset+128], dtype=torch.float64)
            m = optics.metrics(endpoint)
            edges.append(m['difference'][:, 1:].square().mean(-1).sqrt().numpy())
            validity.append(m['valid'].numpy())
    wide49 = np.c_[reported[:, 0], np.concatenate(edges)]
    valid49 = good & np.concatenate(validity)
    with np.load(HERE/'native_edges6000.npz') as z:
        assert np.array_equal(z['original_ids'], ids)
        narrow49 = z['fields'].copy()
        valid_narrow = z['valid'].copy()
    rows = []
    for label, f, valid in [('inner_49nodes', narrow49, valid_narrow),
                          ('outer_49nodes', wide49, valid49), ('outer_audit_grid', reported, good)]:
        rows.append(dict(label=label, fullfield_passes=int((valid & (f.max(1)<.07)).sum()),
            invalid=int((~valid).sum()), per_field_mean=f[valid].mean(0).tolist(),
            per_field_passes=((f<.07)&valid[:, None]).sum(0).tolist()))
    result = dict(count=len(q), epoch=epoch, same_commands=True, same_center_metric=True,
        threshold_waves=.07, edge_metric='DIFF', rows=rows, optimizer_updates=0,
        network_inferences=0, new_optical_states=len(q), seconds=time.perf_counter()-start,
        cuda_initialized=torch.cuda.is_initialized(), training_remains_paused=(BASE/'nikon_large_debug_20261025/STOP').exists())
    assert not result['cuda_initialized'] and result['training_remains_paused']
    (HERE/'same_nodes_field_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
