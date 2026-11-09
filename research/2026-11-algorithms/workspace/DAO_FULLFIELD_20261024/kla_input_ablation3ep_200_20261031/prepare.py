"""Freeze input ablation and reuse data/control/evaluator without replay."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'kla_loss_ablation3ep_200_20261031'
sys.path.insert(0, str(ROOT.parent))
from durable_io import atomic_json
from input_network import ARMS

if (ROOT/'protocol.json').exists():
    raise SystemExit('Already prepared; preserve existing protocol.')
for name in ('selection.npz', 'baseline200.npz', 'control200_grid65.npz', 'bounded_control200.npz'):
    shutil.copy2(PREVIOUS/name, ROOT/name)
old = json.loads((PREVIOUS/'protocol.json').read_text(encoding='utf-8'))
keep = ('initial_checkpoint', 'initial_version', 'control_checkpoint', 'control_epoch',
        'training_samples', 'training_per_cohort', 'training_selection_source',
        'test_samples', 'test_per_cohort', 'selection_seed', 'test_scope', 'selection_ids',
        'optimizer', 'lr', 'weight_decay', 'clip_grad_norm', 'batch', 'microbatch',
        'training_grid', 'network_dtype', 'optics_dtype', 'tf32', 'bn', 'model_seed',
        'order_seed_rule', 'primary_scoring', 'secondary_scoring')
p = {k: old[k] for k in keep}
p.update(task='KLA observed-field count and explicit information representation ablation',
    arms=ARMS, epochs_per_arm=3, updates_per_arm=282, total_new_updates=1128,
    reused_control='Original full9field network, exact archived bounded-loss epoch3,120/200; no retraining or reevaluation',
    initial200_passes=111,
    old_loss='mean_fields(ell/(1+ell)), ell=log1p((PT-WRMS/0.07)^2) +0.05 inverse-pose SmoothL1 +10 original mechanical penalty; original invalid/path penalty',
    loss_changed=False,
    masking='Discarded channels replaced by per-field nominal calibration before every CNN/raw-token/attention branch; same train and inference mask',
    field_indices='0..8 correspond to original meridional angles from-0.132 to+0.132; center4, extremes0/8',
    adapter='2304->128->128->40 MLP with two GELU; zero last layer; additive pre-tanh logits; same size/init for raw and difference arms',
    raw_adapter_input='Original per-field residual (I-I_nominal)/1024,8x8averagepool to16x16 per field',
    difference_adapter_input='Center residual plus signed(field residual-center residual) for eight other fields; identical pooling, same dimensions',
    information_scope='Difference features are derived from the SAME original packet. They add no new physical measurements; raw+diff comparison tests representation, not sensor sufficiency.',
    training='All existing weights and adapter weights trainable; frozen BN running stats, same data/order/optimizer and3epoch budget',
    final_checkpoint='Exact epoch3, no epoch selection, no per-epoch validation',
    input_only_interface='Network receives noisy camera images and fixed calibration only; no true pose, wavefront, PSF, Q or case ID',
    new_acquisitions=0, inference='One packet, one prediction, no loop',
    automatic_full3000=False, automatic_further_training=False, automatic_deployment=False,
    attribution_limits=[
        'Masked1/3field variants warm-start a full9field model; three-epoch adaptation and input distribution shift affect results.',
        'Adapter comparisons versus the original alter capacity; raw-adapter versus diff-adapter holds added capacity and initialization fixed.',
        'No new independent sensing channel is tested, so this cannot prove physical information sufficiency.',
        'One seed,3000training subset and reused200selection cases; not fresh independent validation.'])
atomic_json(ROOT/'protocol.json', p)

source = (PREVIOUS/'train_arm.py').read_text(encoding='utf-8')
source = source.replace('from network import restore', 'from input_network import restore')
source = source.replace("restore(protocol['initial_checkpoint'])", "restore(protocol['initial_checkpoint'], arm=ARM)")
source = source.replace("epoch=epoch, arm=ARM, source_version=", "epoch=epoch, arm=ARM, input_ablation_arm=ARM, source_version=")
source = source.replace('# Only the optical loss changes relative to the archived control.', '# Optical loss and auxiliary coefficients are exactly the archived control.')
source = source.replace("'One seed and three epochs; loss scale and auxiliary balance change with optical loss'",
    "'One seed and three epochs; masked inputs warm-start the full-field model'")
source = source.replace("'A single loss ablation cannot alone establish the unique cause of the KLA bottleneck'",
    "'Same-packet feature changes cannot establish physical measurement sufficiency'")
source = source.replace('KLA loss three-epoch ablation;', 'KLA input three-epoch ablation;')
source = source.replace('Three-epoch direct-WRMS ablation;', 'Three-epoch old-loss input ablation;')
(ROOT/'train_arm.py').write_text(source, encoding='utf-8')

source = (PREVIOUS/'run_suite.py').read_text(encoding='utf-8')
source = source.replace('six-arm', 'four-arm').replace('total_arms=6', 'total_arms=4')
source = source.replace('KLA loss ablation:', 'KLA input ablation, old loss fixed:')
source = source.replace("('psf_l2', 'psf_prediction'), ('psf_l2', 'psf_q2000'),\n            ('psf_prediction', 'psf_prediction_q2000'), ('psf_q2000', 'psf_prediction_q2000')",
    "('center_only', 'three_fields'), ('nine_raw_adapter', 'nine_diff_adapter')")
(ROOT/'run_suite.py').write_text(source, encoding='utf-8')
print(json.dumps(dict(prepared=True, new_arms=list(ARMS), reused_control_passes=120,
                     epochs_each=3, train=3000, test=200, loss_changed=False)))
