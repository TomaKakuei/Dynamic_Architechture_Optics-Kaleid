from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent
queue=json.loads((ROOT/'state.json').read_text())
worker=json.loads((ROOT/queue['arm']/'state.json').read_text())
assert worker['stage']=='training' and worker['updates']>0
note=(f"LATEST THREE KLA CANDIDATES2026-11-01 RUNNING: queuePID{queue['pid']},workerPID{worker['pid']}; "
      f"{queue['arm']} verified{worker['updates']}/282updates. "
      "Oldloss fixed; spatial4x4readout3000x3,model-render residual corrector3000x3,full27000x3. "
      "Same200test, original control120/200 reused. One acquisition; residual candidate adds model computation, not observed feedback. "
      "Native-camera cache precedes residual training; full-data has9x updates. Read kla_three_improvements_20261101/HANDOFF.txt, "
      "state/arm/cache logs. InitialEST80-110min; no notification/full3000/deployment. Prior failed candidates preserved; Olympus locked.\n\n")
path=ROOT.parent/'HANDOFF.txt'
prior=path.read_text(encoding='utf-8')
if not prior.startswith('LATEST THREE KLA CANDIDATES2026-11-01 RUNNING:'):
    path.write_text(note+'--- Historical context retained ---\n'+prior,encoding='utf-8')
(ROOT/'startup_check.json').write_text(json.dumps(dict(queue=queue,worker=worker),indent=2),encoding='utf-8')
print(json.dumps(dict(queue_pid=queue['pid'],worker_pid=worker['pid'],updates=worker['updates'],seconds_per_update=worker['seconds_per_update'])))
