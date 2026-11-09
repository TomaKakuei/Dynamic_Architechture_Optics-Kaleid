from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
queue = json.loads((ROOT/'state.json').read_text(encoding='utf-8'))
worker = json.loads((ROOT/queue['arm']/'state.json').read_text(encoding='utf-8'))
assert worker['stage'] == 'training' and worker['updates'] > 0
note = (f"LATEST KLA INPUT ABLATION2026-10-31 RUNNING: queuePID{queue['pid']},workerPID{worker['pid']}; "
        f"verified{worker['updates']}/282updates in {queue['arm']}. "
        "Old bounded-WRMS loss UNCHANGED. Four new3epoch arms: center1field,3fields,9+rawadapter,9+signeddiffadapter; "
        "same3000train/200test, original9field control120/200 reused. "
        "Same one noisy packet; differences are representation, not added physical sensing. "
        "Read kla_input_ablation3ep_200_20261031/HANDOFF.txt,state.json and worker logs. "
        "InitialEST30-35min, no notification/full3000/deployment. Olympus stays locked.\n\n")
path = ROOT.parent/'HANDOFF.txt'
prior = path.read_text(encoding='utf-8')
if not prior.startswith('LATEST KLA INPUT ABLATION2026-10-31 RUNNING:'):
    path.write_text(note+'--- Historical handoff retained ---\n'+prior, encoding='utf-8')
(ROOT/'startup_check.json').write_text(json.dumps(dict(queue=queue,worker=worker), indent=2), encoding='utf-8')
print(json.dumps(dict(queue_pid=queue['pid'],worker_pid=worker['pid'],updates=worker['updates'],
    seconds_per_update=worker['seconds_per_update'])))
