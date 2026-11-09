"""Interpret committed metrics only; no training, inference or optical replay."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from durable_io import atomic_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


r = read(ROOT/'result.json')
assert r['complete'] and r['completed_arms'] == 6
training = {a: read(ROOT/a/'training_complete.json') for a in r['arms']}
assert all(t['epochs'] == 3 and t['updates'] == 282 for t in training.values())
now = datetime.now(timezone.utc).isoformat()
summary = dict(updated=now, status='completed_and_interpreted',
    method='Read saved result/history only; no independent replay',
    completed_arms=6, total_new_updates=sum(t['updates'] for t in training.values()),
    baseline=r['baseline'], bounded_control=r['bounded_control'],
    rows={a: x['candidate'] | dict(gained_vs_control=x['direct_vs_control']['gained'],
              lost_vs_control=x['direct_vs_control']['lost']) for a, x in r['arms'].items()},
    effective_training_seconds=sum(t['active_train_seconds'] for t in training.values()),
    effective_evaluation_seconds=sum(a['evaluation_active_seconds'] for a in r['arms'].values()),
    component_pairs={k: dict(gained=len(v['gained']), lost=len(v['lost']), net=v['net'])
                     for k, v in r['component_comparisons'].items()},
    direct_wrms_reduction_vs_control_percent=100*(1-r['arms']['direct_wrms']['candidate']['wrms_mean']/r['bounded_control']['wrms_mean']),
    gradient_clipping={a: dict(clipped=t['clipped_updates'], updates=t['updates'],
        mean_preclip_norm=t['mean_grad_norm']) for a, t in training.items()},
    interpretation=[
        'No new objective exceeds the archived same-budget bounded-WRMS control on these200cases.',
        'Direct WRMS lowers mean WRMS and raises mean Q, but loses3 net threshold recoveries; continuous quality and threshold success are not interchangeable.',
        'Adding image consistency reduces PSF recovery101->79 and PSF+Q recovery104->87; Q partially recovers79->87 with image consistency.',
        'Image-loss composite training loss decreases while training WRMS increases; consistent with a competing objective, not proof of its unique cause.',
        'The original successful model already optimized transformed WRMS; the evidence does not support a simple omitted-WRMS/PSF/Q explanation.',
        'One seed, reused validation, fixed unit coefficients and stochastic camera derivatives limit causal/generalization claims. All282updates in every new arm were clipped.',
        'No candidate adopted. Preserve all checkpoints and the previously evaluated full3000 incumbent.'],
    image_loss_definition='Expected original-camera image at inferred initial pose -command minus existing noisy acquisition; no true pose input to image loss',
    inference='One packet, one network prediction; no loop', new_acquisitions=0,
    remaining_seconds=0, automatic_followup=False)
atomic_json(ROOT/'assistant_review.json', summary)
lines = [
    'KLA loss ablation COMPLETE and interpreted. Finished2026-10-31 22:37 PDT.',
    'Six new arms, each3epochs/282updates; reused3000train and fixed200validation.',
    'Initial selected full3000 incumbent on200:111/200=55.5%.',
    'Same-budget bounded-WRMS epoch3 control:120/200=60.0%.',
    'One packet and one neural prediction throughout; original Q2000>=.45 and validity retained.',
    'Reused model-selection validation, not an independent test. All routes invalid0.', '',
    'Loss | passes/200 | percent | mean WRMS waves | gains/losses vs bounded control',
]
for arm, x in r['arms'].items():
    a, p = x['candidate'], x['direct_vs_control']
    lines.append(f"{arm} | {a['passes']}/200 | {100*a['rate']:.1f} | {a['wrms_mean']:.8f} | +{p['gained']}/-{p['lost']}")
lines += ['', 'Interpretation:'] + summary['interpretation']
lines += ['', f"Direct-WRMS mean reduction versus control: {summary['direct_wrms_reduction_vs_control_percent']:.3f}%.",
          f"Effective training {summary['effective_training_seconds']:.2f}s; evaluation {summary['effective_evaluation_seconds']:.2f}s.",
          'Image arms use original native camera, fixed calibration and stochastic field/coordinate central differences.',
          'Prior provisional extra pose-MSE was cancelled before affected training; only original weak pose auxiliary remains.',
          'No background work remains for this suite. EST=0. No automatic full3000, extra training or deployment.']
(ROOT/'FINAL_REVIEW.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
for path in [ROOT/'state.json', ROOT/'result.json']:
    x = read(path)
    x['assistant_interpretation'] = 'completed'
    x['assistant_review_file'] = 'assistant_review.json'
    x['assistant_review_updated'] = now
    atomic_json(path, x)
prefix = ('LATEST: COMPLETE6/6 and interpreted. Same200 recovery: bounded-control120, '
          'WRMS117,PSF101,Q109,PSF+image79,PSF+Q104,PSF+image+Q87; allinvalid0. '
          'No new loss exceeds same-budget control. No model adopted, no background, EST=0. '
          'See FINAL_REVIEW.txt and assistant_review.json; no further training/full3000.\n\n')
path = ROOT/'HANDOFF.txt'
prior = path.read_text(encoding='utf-8')
if not prior.startswith('LATEST: COMPLETE6/6 and interpreted.'):
    path.write_text(prefix+'--- Historical launch/correction context ---\n'+prior, encoding='utf-8')
path = ROOT.parent/'HANDOFF.txt'
prior = path.read_text(encoding='utf-8')
heading = 'LATEST KLA LOSS ABLATION COMPLETE2026-10-31: '
if not prior.startswith(heading):
    path.write_text(heading+prefix+'Directory kla_loss_ablation3ep_200_20261031. Olympus stays locked.\n\n'+prior, encoding='utf-8')
print(json.dumps(dict(completed=True, new_updates=summary['total_new_updates'],
    training_seconds=summary['effective_training_seconds'], evaluation_seconds=summary['effective_evaluation_seconds'],
    wrms_reduction_percent=summary['direct_wrms_reduction_vs_control_percent'])))
