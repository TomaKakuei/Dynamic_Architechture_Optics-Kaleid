"""Register a fixed 200-case paired experiment; no optical computation."""
from pathlib import Path
import json
import sys
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
DAO = OUT.parent
PREVIOUS = DAO / 'kla_objective_alignment_20261028/training_compare'
CONTROL = DAO / 'kla_unsaturated_loss_20261028/training_compare'
sys.path.insert(0, str(DAO))
from durable_io import atomic_json

if (OUT / 'protocol.json').exists():
    raise SystemExit('Already prepared; use the registered selection.')
with np.load(CONTROL / 'selection.npz') as z:
    train, pool = z['train'].copy(), z['valrows'].copy()
with np.load(PREVIOUS / 'selection.npz') as z:
    assert np.array_equal(train, z['train'])
checkpoint = torch.load(CONTROL / 'continued_bounded/candidate.pt', map_location='cpu', weights_only=False)
assert checkpoint['epoch'] == checkpoint['selected_epoch'] == 3
assert checkpoint['source_version'] == 'old_wfe_stable_epoch4'
assert checkpoint['arm'] == 'continued_bounded'
prior = checkpoint['protocol']
assert prior['lr'] == 3e-5 and prior['batch'] == 32 and prior['microbatch'] == 16
assert prior['training_grid'] == 65 and prior['training_samples'] == 3000
rng = np.random.default_rng(20261031)
rows = np.concatenate([rng.choice(pool[pool // 1000 == c], n, replace=False)
                       for c, n in enumerate((67, 67, 66))])
assert len(rows) == len(np.unique(rows)) == 200
with np.load(PREVIOUS / 'selected_validation3000_grid129.npz') as z:
    baseline = {k: z[k][rows].copy() for k in ('command', 'wrms', 'fields', 'support', 'indices')}
with np.load(PREVIOUS / 'selected_refined_scores.npz') as z:
    baseline.update(score=z['score'][rows].copy(), valid=z['valid'][rows].copy())
with np.load(CONTROL / 'continued_bounded/epoch3_validation600.npz') as z:
    lookup = {int(r): i for i, r in enumerate(z['rows'])}
    pos = np.array([lookup[int(r)] for r in rows])
    control = {k: z[k][pos].copy() for k in ('command', 'score', 'valid', 'indices')}
assert np.array_equal(baseline['indices'], control['indices'])
assert not np.intersect1d(train, baseline['indices']).size
np.savez_compressed(OUT / 'selection.npz', train=train, rows=rows, indices=baseline['indices'])
np.savez_compressed(OUT / 'baseline200.npz', **baseline, rows=rows)
np.savez_compressed(OUT / 'control200_grid65.npz', **control, rows=rows)
protocol = dict(
    task='KLA direct WRMS versus archived bounded-WRMS, fixed three-epoch ablation',
    initial_checkpoint=str(PREVIOUS / 'old_wfe_stable/candidate.pt'),
    initial_version='old_wfe_stable_epoch4; full original3000 one-shot1684',
    control_checkpoint=str(CONTROL / 'continued_bounded/candidate.pt'),
    control_epoch=3, control_reuse='Reuse the exact archived epoch3 commands; no retraining or neural inference',
    epochs=3, updates_per_epoch=94, new_optimizer_updates=282,
    training_samples=3000, training_per_cohort=1000,
    training_selection_source=str(CONTROL / 'selection.npz'),
    test_samples=200, test_per_cohort=[67, 67, 66], selection_seed=20261031,
    test_scope='Outcome-independent draw from the old600 model-selection validation, reused and not independent',
    selection_ids=baseline['indices'].tolist(),
    control_optical_loss='mean_fields(ell/(1+ell)), ell=log1p((PT-WRMS/0.07)^2)',
    candidate_optical_loss='mean_fields(PT-WRMS/0.07), linear normalized WRMS; no log, square, saturation or nominal-phase subtraction',
    wrms_definition='RMS wavefront in waves over surviving uniformly sampled pupil; remove piston and tip/tilt only, retain defocus at fixed original image plane',
    auxiliary_loss='unchanged: 0.05 inverse-pose SmoothL1 + 10 mechanical penalty, original invalid/path penalty',
    attribution_limit='Optical loss replacement also changes gradient scale and relative field/case/auxiliary weighting; no unique-cause claim from one seed',
    optimizer='fresh AdamW', lr=3e-5, weight_decay=1e-4, clip_grad_norm=5,
    batch=32, microbatch=16, training_grid=65,
    network_dtype='float32', optics_dtype='float64', tf32=False,
    bn='Frozen running statistics; affine weights trainable',
    model_seed=2026102819, order_seed_rule='2026102900+epoch, identical to archived bounded continuation',
    final_checkpoint='exact epoch3, no best-epoch selection and no per-epoch validation',
    primary_scoring='Original Q2000>=0.45 plus complete-path geometry and >=98% pupil support; grid129 with original grid257 borderline/disagreement refinement',
    secondary_scoring='mean and worst-field PT-WRMS at common grid129; casewise WRMS/Q agreement',
    baseline_reuse='Saved full3000 refined score/validity and grid129 WRMS, subset only',
    inference='One archived noisy packet, one neural prediction, no true state in network, no loop',
    automatic_full3000=False, automatic_further_training=False, automatic_deployment=False,
    new_acquisitions=0, data_generation=False)
atomic_json(OUT / 'protocol.json', protocol)
print(json.dumps(dict(prepared=True, control_epoch=checkpoint['epoch'], training_samples=len(train),
                     test_samples=len(rows), baseline_passes=int((baseline['valid'] & (baseline['score'] >= .45)).sum()))))
