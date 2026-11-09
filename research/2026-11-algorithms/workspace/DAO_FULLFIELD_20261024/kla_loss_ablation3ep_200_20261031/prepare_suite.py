"""Reuse the registered data and generate the bounded sequential suite runner."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
DRAFT = ROOT.parent / 'kla_wrms_ablation3ep_200_20261031'
sys.path.insert(0, str(ROOT.parent))
from durable_io import atomic_json
from losses import ARMS

if (ROOT / 'protocol.json').exists():
    raise SystemExit('Already prepared; do not replace a running suite.')
for name in ('selection.npz', 'baseline200.npz', 'control200_grid65.npz'):
    shutil.copy2(DRAFT/name, ROOT/name)
p = json.loads((DRAFT/'protocol.json').read_text(encoding='utf-8'))
p.update(task='KLA three-epoch loss ablation, six new arms plus reused same-budget control',
         arms=ARMS, epochs_per_arm=3, new_optimizer_updates=6*282,
         planned_training_passes=6*3, candidate_optical_loss='See loss_definitions and arms',
         prediction_difference_interpretation='Provisional: normalized command prediction minus inverse true pose, used in training only',
         prediction_difference_status='User clarification asked; default pose-difference assumption stated before launch',
         loss_definitions={
             'w': 'mean_fields(PT-WRMS/0.07)',
             'p': 'mean_fields(sum_pixels((unit_flux_PSF(endpoint)-unit_flux_PSF(nominal))^2)/sum_pixels(unit_flux_PSF(nominal)^2))',
             'q': '((Q2000(endpoint)-Q2000(nominal))/Q2000(nominal))^2, same original scalar Q definition',
             'd': 'mean_40_coordinates((normalized_command+normalized_initial_pose)^2)'},
         combinations='Fixed unit coefficients for included terms; additive, no per-case weighting or tuning on200',
         common_auxiliary='All arms retain 0.05 SmoothL1 inverse pose +10 mechanical penalty; D is additional MSE supervision, not the first use of pose labels',
         psf_history='Earlier PSF Hellinger arm failed; new P is relative squared pixel L2, not a repeat',
         sequential=True, per_epoch_validation=False,
         automatic_full3000=False, automatic_deployment=False)
atomic_json(ROOT/'protocol.json', p)

# Preserve the original prepared-only WRMS draft, with an explicit supersession record.
atomic_json(DRAFT/'SUPERSEDED.json', dict(reason='User expanded loss ablation before launch',
    successor=str(ROOT), training_started=False))
source = (DRAFT/'run.py').read_text(encoding='utf-8')
source = source.replace('OUT = Path(__file__).resolve().parent\nDAO = OUT.parent',
    "ROOT = Path(__file__).resolve().parent\nARM = sys.argv[1]\nOUT = ROOT / ARM\nOUT.mkdir(exist_ok=True)\nDAO = ROOT.parent\nfrom losses import OpticalLoss, ARMS\nassert ARM in ARMS")
source = source.replace("(OUT / 'protocol.json')", "(ROOT / 'protocol.json')")
for name in ('selection.npz', 'control200_grid65.npz', 'baseline200.npz'):
    source = source.replace(f"OUT / '{name}'", f"ROOT / '{name}'")
source = source.replace("return (OUT / 'STOP_REQUESTED.json').exists()",
    "return (OUT / 'STOP_REQUESTED.json').exists() or (ROOT / 'STOP_REQUESTED.json').exists()")
source = source.replace("geometry, optic = Geometry(), Optics(grid=65, device='cuda')",
    "geometry, optic = Geometry(), Optics(grid=65, device='cuda')\n    objective = OpticalLoss(optic, ARM)")
source = source.replace("arm='direct_wrms'", "arm=ARM")
source = source.replace("physical = (wrms/.07).mean(-1)",
    "physical, loss_components = objective(phase, mask, wrms, cmd, target)")
source = source.replace("network_dtype='float32', gpu_peak_gib=", "arm=ARM, last_microbatch_components=loss_components, network_dtype='float32', gpu_peak_gib=")
source = source.replace("candidate = evaluate('direct_wrms', command, metrics(optic, command, 16))",
    "candidate = evaluate('candidate', command, metrics(optic, command, 16))")
source = source.replace("target = OUT / f'{name}200.npz'",
    "target = (ROOT if name == 'bounded_control' else OUT) / f'{name}200.npz'")
source = source.replace("direct_wrms=summarize(candidate)", "candidate=summarize(candidate), arm=ARM")
source = source.replace("('baseline', 'bounded_control', 'direct_wrms')", "('baseline', 'bounded_control', 'candidate')")
source = source.replace("result['direct_wrms']['passes']", "result['candidate']['passes']")
source = source.replace("KLA direct-WRMS three-epoch ablation;", "KLA loss three-epoch ablation;")
source = source.replace("A direct-WRMS result cannot alone establish", "A single loss ablation cannot alone establish")
(ROOT/'train_arm.py').write_text(source, encoding='utf-8')
print(json.dumps(dict(prepared=True, arms=list(ARMS), cases=200, epochs_each=3)))
