from pathlib import Path
from datetime import datetime,timezone
import json,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from durable_io import atomic_json
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
queue=read(ROOT/'state.json')
worker=read(ROOT/'model_residual/state.json')
assert worker['stage']=='training' and worker['updates']>154
r=read(ROOT/'RESUME_RECORD.json')
r.update(resume_verified=datetime.now(timezone.utc).isoformat(),new_queue_pid=queue['pid'],
    new_worker_pid=worker['pid'],verified_resumed_updates=worker['updates'],
    remaining_estimate_minutes=[65,80])
atomic_json(ROOT/'RESUME_RECORD.json',r)
atomic_json(ROOT/'resume_startup_check.json',dict(queue=queue,worker=worker))
text=('Partial results, interpreted2026-11-01; suite not complete.\n'
      'Spatial readout115/200=57.5%,same3000x3old-loss control120/200=60%.\n'
      'Spatial gained5/lost10; invalid0. No improvement, no deployment.\n'
      'Model-residual cache3200complete,zero rendering fallbacks; no full metric yet.\n'
      'Old processes disappeared after154committed updates; empty stderr,no stop marker.\n'
      'Cause unknown. Original evidence and checkpoint archived; no source/numeric changes.\n'
      f'Resumed queue{queue["pid"]},worker{worker["pid"]}; confirmed{worker["updates"]}/282updates.\n'
      'Full27000x3not started; it remains queued after residual evaluation.\n'
      'Remaining65-80min effective; downtime excluded. No notification configured.\n')
(ROOT/'INTERIM_REVIEW.txt').write_text(text,encoding='utf-8')
print(json.dumps(dict(queue_pid=queue['pid'],worker_pid=worker['pid'],updates=worker['updates'])))
