"""Interpret saved input-ablation evidence without inference or scoring replay."""
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
assert r['complete'] and r['completed_arms'] == 4
training = {a: read(ROOT/a/'training_complete.json') for a in r['arms']}
assert all(t['epochs'] == 3 and t['updates'] == 282 for t in training.values())
now = datetime.now(timezone.utc).isoformat()
report = dict(status='completed_and_interpreted', updated=now,
    evidence='Read saved metrics and training histories only; no inference, optical replay or new training',
    control=r['bounded_control'], initial=r['baseline'],
    arms={a: x['candidate'] | dict(gains_vs_control=x['direct_vs_control']['gained'],
         losses_vs_control=x['direct_vs_control']['lost']) for a, x in r['arms'].items()},
    paired={k: dict(gained=len(v['gained']), lost=len(v['lost']), net=v['net'])
            for k, v in r['component_comparisons'].items()},
    effective_training_seconds=sum(t['active_train_seconds'] for t in training.values()),
    effective_evaluation_seconds=sum(x['evaluation_active_seconds'] for x in r['arms'].values()),
    epoch_training_wrms={a: [h['train_means'][2] for h in t['history']] for a, t in training.items()},
    conclusion=[
        'No new input variant beats the original nine-field bounded-loss epoch3 control120/200.',
        'Raw adapter116/200 adds4 and loses8 against control. Difference adapter105/200 adds5 and loses20.',
        'At matched added capacity and identical initial prediction, signed-difference features lose11net cases versus raw features (4gains15losses).',
        'The1/3field masks yield0/1passes but are not clean information-theoretic evidence. They warm-start nine-field weights with frozen BN and large input distribution changes, followed by only three epochs.',
        'Masked training WRMS is still declining strongly across epochs: center6.787->3.248->2.034,three4.115->1.124->0.526waves. Adaptation and old-loss saturation remain confounded with information removal.',
        'All200 endpoints are valid for every route; losses here are optical quality failures, not invalid-case exclusions.',
        'No independent observation was added. This experiment measures channel-removal adaptation and same-packet representation, and does not settle whether the original measurement contains enough physical information.',
        'One seed,3000training subset and reused200selection cases; do not claim an independent test or full3000 ranking.',
        'Keep the incumbent. No candidate adoption, no automatic follow-up.'],
    remaining_seconds=0, queue_and_final_worker_exited=True,
    new_acquisitions=0, loss_changed=False, automatic_followup=False)
atomic_json(ROOT/'assistant_review.json', report)
lines = [
    'KLA input ablation COMPLETE and interpreted. Finished2026-10-31 23:34 PDT.',
    'Old bounded-WRMS,pose and mechanical loss unchanged. Four new arms,3epochs/282updates each.',
    'Same3000training and200reused selection cases; one original packet,one prediction,no loop.',
    'Original full9field matched-budget control120/200=60.0%; initial checkpoint111/200=55.5%.',
    'All routes have0invalid endpoints. Queue and final worker have exited; EST=0.', '',
    'Input | passes/200 | percent | mean WRMS waves | gains/losses vs9field control']
for arm, x in r['arms'].items():
    a, p = x['candidate'], x['direct_vs_control']
    lines.append(f"{arm} | {a['passes']}/200 | {100*a['rate']:.1f} | {a['wrms_mean']:.8f} | +{p['gained']}/-{p['lost']}")
lines += ['', 'Interpretation:']+report['conclusion']
lines += ['', f"Effective training{report['effective_training_seconds']:.2f}s; evaluation{report['effective_evaluation_seconds']:.2f}s.",
    'Original981584parameters; both adapters1298296parameters. No extra independent sensing channels.',
    'Original weights, failed candidates and all source evidence preserved. No full3000, deployment or next training queued.']
(ROOT/'FINAL_REVIEW.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
for path in [ROOT/'state.json', ROOT/'result.json']:
    x = read(path)
    x.update(assistant_interpretation='completed', assistant_review_file='assistant_review.json',
             assistant_review_updated=now)
    atomic_json(path, x)
for arm in r['arms']:
    path = ROOT/arm/'state.json'
    x = read(path)
    x.update(assistant_interpretation='completed_in_suite', assistant_review_file='../assistant_review.json')
    atomic_json(path, x)
note = ('LATEST INPUT ABLATION COMPLETE: old loss fixed, same200. Full9control120,center1field0,'
        'threefields1,9+rawadapter116,9+diffadapter105; allinvalid0. No new winner. '
        'Masked results confound information removal with warm-start/input-distribution adaptation; '
        'no claim of physical information insufficiency. No background,EST=0,no next training/deployment. '
        'See FINAL_REVIEW.txt and assistant_review.json in kla_input_ablation3ep_200_20261031.\n\n')
for path in (ROOT/'HANDOFF.txt', ROOT.parent/'HANDOFF.txt'):
    old = path.read_text(encoding='utf-8')
    if not old.startswith('LATEST INPUT ABLATION COMPLETE:'):
        path.write_text(note+'--- Historical context retained ---\n'+old, encoding='utf-8')
print(json.dumps(dict(completed=True, total_new_updates=4*282,
    effective_seconds=report['effective_training_seconds']+report['effective_evaluation_seconds'],
    new_winner=False)))
