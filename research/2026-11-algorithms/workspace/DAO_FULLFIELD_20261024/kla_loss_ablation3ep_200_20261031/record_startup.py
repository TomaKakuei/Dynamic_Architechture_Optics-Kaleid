"""Update only local handoff from the running experiment's actual state."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
queue = json.loads((ROOT/'state.json').read_text(encoding='utf-8'))
worker = json.loads((ROOT/queue['arm']/'state.json').read_text(encoding='utf-8'))
assert worker['stage'] == 'training' and worker['updates'] > 0
note = (f"LATEST2026-10-31: KLA six-loss3epoch ablation RUNNING; queuePID{queue['pid']}, "
        f"first workerPID{worker['pid']}, verified {worker['updates']}/282 updates. "
        "Same3000train/fixed200validation, FP32 network, one-shot/no loop. "
        "New WRMS,PSF-L2,Q2000-gap,PSF+prediction,PSF+Q,PSF+prediction+Q; "
        "prediction difference provisionally inverse-pose MSE (clarification asked). "
        "Same-budget bounded control reused. See kla_loss_ablation3ep_200_20261031/HANDOFF.txt, "
        "state.json and arm/state.json. Serial resumable, initialEST45-60min; "
        "no full3000/deployment/notification. Old WRMS-only prepared folder superseded and never launched. "
        "Olympus remains locked.\n\n")
parent = ROOT.parent/'HANDOFF.txt'
prior = parent.read_text(encoding='utf-8')
if not prior.startswith('LATEST2026-10-31: KLA six-loss3epoch'):
    parent.write_text(note+'--- Historical handoff retained ---\n'+prior, encoding='utf-8')
(ROOT/'startup_check.json').write_text(json.dumps(dict(queue=queue, worker=worker), indent=2), encoding='utf-8')
print(json.dumps(dict(queue_pid=queue['pid'], worker_pid=worker['pid'], updates=worker['updates'],
                     seconds_per_update=worker['seconds_per_update'])))
