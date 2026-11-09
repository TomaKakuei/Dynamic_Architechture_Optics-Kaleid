"""One user-requested 100-case estimate from the paused checkpoint; no replay."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from predict import load_model, run


def main():
    paused = json.loads((run.HERE/'state.json').read_text())
    if paused['status'] != 'paused':
        raise RuntimeError('Training must save its checkpoint and pause before this estimate')
    out = run.HERE/'pause_sample100'
    out.mkdir(exist_ok=True)
    if (out/'result.json').exists():
        print((out/'result.json').read_text())
        return
    with np.load(run.HERE/'data/metadata.npz') as z:
        data = {k:z[k].copy() for k in z.files}
    validation = np.flatnonzero(~data['train'])
    selection_seed = 2026102601
    indices = np.sort(np.random.default_rng(selection_seed).choice(validation, size=100, replace=False))
    run.exp.atomic(out/'selection.json', dict(count=100, selection_seed=selection_seed,
        selection='Uniform without replacement from original 6000 validation rows, before prediction',
        indices=indices.tolist(), original_ids=data['ids'][indices].tolist(),
        cohorts=data['cohort'][indices].tolist(), checkpoint='resume.pt',
        paused_epoch=paused['epoch'], paused_batch=paused['batch']))
    raw = np.load(run.HERE/'data/raw.npy', mmap_mode='r')
    tick = run.exp.active_time()
    model = load_model(run.HERE/'resume.pt')
    commands, fields, validity = [], [], []
    with torch.no_grad():
        for first in range(0, len(indices), run.MICRO):
            ix = indices[first:first+run.MICRO]
            command = model.predict_packet(np.array(raw[ix], copy=True))
            # The hidden initial pose appears only in simulator endpoint scoring.
            endpoint = torch.as_tensor(data['q'][ix], device='cuda')+command.double()
            signal, valid = run.packet(model.native_optics, endpoint, per_field_valid=True)
            commands.append(command.cpu().numpy())
            fields.append(signal.square().mean(-1).sqrt().cpu().numpy())
            validity.append(valid.cpu().numpy())
    command, w, valid = [np.concatenate(x) for x in (commands,fields,validity)]
    def score(columns):
        good = valid[:, columns].all(1) & np.isfinite(w[:, columns]).all(1)
        passed = good & (w[:, columns].max(1)<.07)
        return dict(passes=int(passed.sum()),count=len(w),recovery_rate=float(passed.mean()),
                    invalid=int((~good).sum()))
    result = dict(status='completed', samples=100, validation_population=6000,
        selection_seed=selection_seed, paused_epoch=paused['epoch'],paused_batch=paused['batch'],
        optimizer_updates=(paused['epoch']-1)*219+paused['batch'],
        threshold_waves=.07, measurement_acquisitions=1, model_refinements=4,
        measured_feedback=False, truth_inputs_to_network=False, independent_review=False,
        center=score([0]), original_three_fields=score([0,3,4]),
        center_and_each_radius={str(radius):score([0,2*i+1,2*i+2])
                               for i,radius in enumerate((.05,.1,.15))},
        all_seven_fields=score(list(range(7))),fields_mm=list(run.FIELDS),
        per_field_passes=((w<.07)&valid).sum(0).tolist(),
        per_field_mean=[float(w[valid[:,i],i].mean()) if valid[:,i].any() else None for i in range(7)],
        cohort_counts={str(c):int((data['cohort'][indices]==c).sum())
                       for c in np.unique(data['cohort'][indices])},
        active_seconds=run.exp.active_time()-tick,updated=run.exp.stamp())
    np.savez(out/'predictions.npz',indices=indices,original_ids=data['ids'][indices],
             command=command,fields=w,field_valid=valid,cohort=data['cohort'][indices])
    run.exp.atomic(out/'result.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
