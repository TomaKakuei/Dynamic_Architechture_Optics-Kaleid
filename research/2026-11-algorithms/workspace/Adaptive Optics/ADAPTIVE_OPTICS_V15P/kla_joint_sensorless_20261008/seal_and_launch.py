"""Seal successful checks, estimate remaining work, launch hidden coordinator."""
from common import *
import sys,subprocess,psutil,math
def run():
 checks={}
 for name in ('optics_check.json','camera_check.json','camera_high_bandwidth_check.json','training_preflight.json'):
  p=HERE/name;assert json.loads(p.read_text())['status']=='pass';checks[name]=hashlib.sha256(p.read_bytes()).hexdigest()
 pre=json.loads((HERE/'training_preflight.json').read_text());assert pre['code']==hashes(),'Training preflight must cover final source'
 protected=[]
 for p in [HERE.parents[1]/'kla_catadioptric_engine.py',HERE.parents[1]/'kla_full_fov_engine.py',HERE.parent/'trepan_restored_joint_20261008/training/candidate.pt',HERE.parent/'nikon_coordinated_center_20261005/summary.json']+list((HERE.parent/'other_paths_differentiable_20261001').glob('kla_module*/candidate.pt')):
  if p.exists():protected.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 seal=dict(created=stamp(),code=hashes(),checks=checks,camera_calibration_sha256=hashlib.sha256((HERE/'camera_calibration.json').read_bytes()).hexdigest(),architecture_sha256=hashlib.sha256((HERE.parent/'canonical_three_phase_universal/models.py').read_bytes()).hexdigest(),protected_files=protected)
 atomic(HERE/'launch_seal.json',seal)
 progress=json.loads((HERE/'generation_progress.json').read_text());remaining=TOTAL-progress['completed'];rate=progress['seconds_per_new_sample'];train=50*(math.ceil(3*TRAIN/32)*pre['steady_step_seconds']+(3*VALIDATION+500)*pre['evaluation_seconds_per_case']);generate=remaining*rate
 estimate=dict(updated=stamp(),generation_hours=generate/3600,training_hours=train/3600,audit_hours_range=[.1,.4],total_hours_range=[(generate+train)*.9/3600+.1,(generate+train)*1.25/3600+.4],basis=dict(pilot_samples=progress['completed'],sampling_seconds_per_sample=rate,batch32_seconds=pre['steady_step_seconds'],evaluation_seconds_per_case=pre['evaluation_seconds_per_case'],training_batches_per_epoch=math.ceil(3*TRAIN/32)),effective_time_only=True,high_resolution_tail='Pilot150 required no CPUfallback; rare later high-bandwidth captures use slower CPU rendering and can increase generation time. Re-estimate from generation_progress.json and per-block render_grids.')
 atomic(HERE/'estimate.json',estimate)
 logs=HERE/'logs';logs.mkdir(exist_ok=True)
 with (logs/'pipeline.log').open('a',encoding='utf-8') as out,(logs/'pipeline.err').open('a',encoding='utf-8') as err:
  p=subprocess.Popen([sys.executable,'-X','utf8','-u',str(HERE/'pipeline.py'),'--workers','6'],cwd=HERE,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
 atomic(HERE/'launch_request.json',dict(pid=p.pid,updated=stamp(),estimate=estimate))
 (HERE.parent/'KLA_LATEST_RUN.txt').write_text(str(HERE)+'\n',encoding='utf-8')
 print(json.dumps(dict(pid=p.pid,estimate=estimate)),flush=True)
if __name__=='__main__':run()
