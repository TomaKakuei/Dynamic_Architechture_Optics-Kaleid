"""Preserve interruption evidence and read the committed optimizer checkpoint."""
from pathlib import Path
from datetime import datetime,timezone
import json,shutil,sys
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from durable_io import atomic_json

tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive=ROOT/('resume_'+tag)
archive.mkdir()
for name in ('state.json','started.json','suite.stdout.log','suite.stderr.log'):
    path=ROOT/name
    if path.exists():shutil.copy2(path,archive/name)
part=archive/'model_residual';part.mkdir()
for name in ('state.json','stdout.log','stderr.log','history.json','resume.pt','cache_complete.json'):
    shutil.copy2(ROOT/'model_residual'/name,part/name)
c=torch.load(ROOT/'model_residual/resume.pt',map_location='cpu',weights_only=False)
updates=(int(c['next_epoch'])-1)*94+int(c['next_batch'])
assert updates==154 and c['next_epoch']==2 and c['next_batch']==60
assert not (ROOT/'STOP_REQUESTED.json').exists()
assert not (ROOT/'model_residual/STOP_REQUESTED.json').exists()
evidence=dict(recorded=datetime.now(timezone.utc).isoformat(),archive=str(archive),
    previous_queue_pid=17756,previous_worker_pid=33024,python_processes_at_check=0,
    interruption='Processes gone; no Python error traceback or stop marker. Cause not established.',
    checkpoint_committed_updates=updates,next_epoch=int(c['next_epoch']),next_batch=int(c['next_batch']),
    checkpoint_active_train_seconds=float(c['active_seconds']),remaining_model_residual_updates=282-updates,
    reused_camera_packets=3200,cache_fallbacks=0,
    completed_spatial_passes=115,comparison_control_passes=120,
    full_data_started=False,full_data_planned_updates=2532,
    source_or_numerical_changes=False,plan='Resume same script/checkpoint/optimizer, skip finished spatial result and camera cache')
atomic_json(ROOT/'RESUME_RECORD.json',evidence)
notes=('INTERRUPTION DETECTED2026-11-01: old queue17756/worker33024 are gone, no traceback/STOP. '
       'Only spatial_head complete115/200 vs120control. model_residual committed154/282updates, '
       'all3200camera contexts complete with0fallback. full_data not started. '
       'Original evidence and154update checkpoint preserved in '+archive.name+'. '
       'Resume same scripts/weights/optimizer; no numerical changes, no cache/spatial replay. '
       'Read resume_started.json/current state for new processes. Remaining estimate65-80min effective '
       '(residual~5-7min plus full27000x3~60-70min). Downtime excluded, cause unknown.\n\n')
for path in (ROOT/'HANDOFF.txt',ROOT.parent/'HANDOFF.txt'):
    prior=path.read_text(encoding='utf-8')
    path.write_text(notes+'--- Earlier context retained ---\n'+prior,encoding='utf-8')
print(json.dumps(evidence))
