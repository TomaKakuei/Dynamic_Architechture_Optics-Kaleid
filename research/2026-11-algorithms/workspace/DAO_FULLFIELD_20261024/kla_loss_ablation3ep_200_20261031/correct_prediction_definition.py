"""Record the user's clarification before either image-loss arm begins."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from durable_io import atomic_json

for name in ('psf_prediction', 'psf_prediction_q2000'):
    if (ROOT/name/'resume.pt').exists() or (ROOT/name/'training_complete.json').exists():
        raise RuntimeError('An affected arm has started; preserve it separately before changing semantics')
path = ROOT/'protocol.json'
old = path.read_text(encoding='utf-8')
(ROOT/'protocol_before_image_clarification.json').write_text(old, encoding='utf-8')
p = json.loads(old)
p.update(prediction_difference_interpretation='USER CONFIRMED: predicted camera image minus actual acquired image',
    prediction_difference_status='Confirmed by user; replaces provisional pose-MSE before affected arms began',
    common_auxiliary='All arms retain original0.05 SmoothL1 inverse pose +10 mechanical; NO additional pose-MSE term',
    image_consistency=dict(
        estimated_pose='-network_command, estimates the initial misalignment; no true pose input',
        target='Existing original noisy9x128x128 ADU packet',
        fixed_loss_scale='MSE in ADU divided by1024^2, matching original input scale; no per-image flux normalization',
        camera='Original fixed diffuser, exposure, PRNU/DSNU, degree24,2048pupil/4096FFT and original4096/8192 bandwidth fallback',
        derivative='Stochastic coordinate central difference h=0.001; random1of40 coordinate multiplied by40; includes complete boundary/carrier/phase response',
        field_sampling='Uniform one of9fields per training example per epoch, same draws across image-loss arms',
        seed='202610310000 + epoch*1000000 + global_case_id*17',
        loss_value='Symmetric average of plus/minus full-camera residual MSE; finite-difference smoothing',
        compute='Two full-camera field renders per training example, no new noisy acquisitions and no extra optimizer steps',
        limitations='Stochastic image-gradient estimator; variance and finite-difference approximation must be distinguished from loss-design effects'))
p['loss_definitions']['d'] = 'E_field[mean_pixels((ExpectedADU(-command)-ObservedADU)^2)/1024^2], stochastic field/coordinate finite-difference estimate'
p['attribution_limit'] += '; image-loss arms additionally use a recorded stochastic camera-gradient estimator'
atomic_json(path, p)
atomic_json(ROOT/'IMAGE_PREDICTION_CORRECTION.json', dict(updated=datetime.now(timezone.utc).isoformat(),
    user_definition='predicted image minus acquired image', affected_arms=['psf_prediction','psf_prediction_q2000'],
    prior_pose_mse_training_updates=0, same_data_and_test200=True, unaffected_training_preserved=True,
    old_eta='45-60min no longer applies to the full suite; image arms need live camera timings'))
note = ('USER CLARIFICATION: D is predicted camera image minus acquired image.\n'
        'Provisional extra pose-MSE cancelled before either affected arm started.\n'
        'D renders the inferred initial pose -command through the original camera;\n'
        'observed noisy ADU is the target. True initial pose is NOT used by D.\n'
        'Original weak0.05 SmoothL1 remains common to all arms.\n'
        'Full native camera raster kept. To bound computation: uniform1/9field\n'
        'and1/40coordinate per example, central h.001,40x gradient,2renders.\n'
        'Same draws across D arms. Stochastic gradient/FD smoothing is an explicit\n'
        'limitation; no image-loss-only causal claim. Protocol has full definition.\n'
        'Each arm still3epochs, same3000train/200test. Original45-60min estimate\n'
        'is superseded; use first D-arm state image_loss timings to estimate.\n\n')
handoff = ROOT/'HANDOFF.txt'
handoff.write_text(note+'--- Original launch context ---\n'+handoff.read_text(encoding='utf-8'), encoding='utf-8')
parent = ROOT.parent/'HANDOFF.txt'
parent.write_text('LATEST KLA clarification: prediction difference = original-camera predicted image minus observed image. '
    'No extra pose-MSE training was run. Six-arm suite continues in kla_loss_ablation3ep_200_20261031; '
    'see IMAGE_PREDICTION_CORRECTION.json and current state. Original ETA superseded for image arms.\n\n'
    + parent.read_text(encoding='utf-8'), encoding='utf-8')
print('Corrected protocol and handoff; no affected arm had started.')
