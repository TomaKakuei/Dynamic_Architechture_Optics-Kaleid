"""Score all saved commands at sensor edges; keep the original center metric."""
from audit_sensor_scope import *


def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    legacy = exp.load_module('paused_full_native_edge_check', exp.ROOT/'nikon_coordinated_center_20261005/job.py')
    native = legacy.NikonTensorOptics(nodes=legacy.nikon35.NODES,
        fields=legacy.nikon35.base.corrected.CONTROL_FIELDS_MM, device='cpu')
    folder = BASE/'nikon_large_debug_20261025/trajectory_4ep'
    epoch = json.loads((folder/'selected.json').read_text())['epoch']
    with np.load(folder/f'validation_{epoch:03d}.npz') as z:
        initial, command, reported, valid, ids = [z[k].copy()
            for k in ('initial', 'command', 'fields', 'valid', 'original_ids')]
    edges, validity = [], []
    with torch.no_grad():
        for start_row in range(0, len(initial), 128):
            q = torch.tensor(initial[start_row:start_row+128], dtype=torch.float64)
            c = torch.tensor(command[start_row:start_row+128], dtype=torch.float64)
            metrics = native.metrics(q+c)
            edges.append(metrics['difference'][:, 1:].square().mean(-1).sqrt().numpy())
            validity.append(metrics['valid'].numpy())
    edges = np.concatenate(edges)
    native_valid = np.concatenate(validity)
    narrow = np.c_[reported[:, 0], edges]
    pass_narrow = valid & native_valid & (narrow.max(1) < .07)
    pass_wide = valid & (reported.max(1) < .07)
    def summary(fields, good):
        return dict(count=len(fields), invalid=int((~good).sum()),
            per_field_mean=fields[good].mean(0).tolist(),
            per_field_passes=((fields < .07) & good[:, None]).sum(0).tolist(),
            fullfield_passes=int((good & (fields.max(1) < .07)).sum()))
    result = dict(status='completed', epoch=epoch, count=len(initial),
        same_saved_commands=True, same_center_absolute_metric=True, edge_metric='DIFF',
        sensor_edges_mm=[-.05, .05], report_edges_mm=[-.1, .1],
        native=summary(narrow, valid & native_valid), reported=summary(reported, valid),
        narrow_pass_wide_fail=int((pass_narrow & ~pass_wide).sum()),
        narrow_fail_wide_pass=int((~pass_narrow & pass_wide).sum()),
        seconds=time.perf_counter()-start, device='cpu', optimizer_updates=0,
        network_inferences=0, new_optical_states=len(initial),
        cuda_initialized=torch.cuda.is_initialized(),
        training_remains_paused=(BASE/'nikon_large_debug_20261025/STOP').exists(),
        scope='Diagnosis only. Official +/-0.1mm criterion and existing recovery results are unchanged.')
    assert not result['cuda_initialized'] and result['training_remains_paused']
    np.savez_compressed(HERE/'native_edges6000.npz', original_ids=ids, fields=narrow,
        valid=valid & native_valid)
    (HERE/'native_edges6000.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
