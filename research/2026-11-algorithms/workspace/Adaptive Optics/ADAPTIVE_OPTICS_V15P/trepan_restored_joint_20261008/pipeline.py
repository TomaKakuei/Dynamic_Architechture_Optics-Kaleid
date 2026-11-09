"""Restart-safe serial generator -> joint50epoch trainer -> independent audit."""
from common import *
import sys,subprocess,traceback,psutil,argparse
def run(workers):
 previous=json.loads((HERE/'started.json').read_text()) if (HERE/'started.json').exists() else {}
 if previous.get('pid') and psutil.pid_exists(previous['pid']):
  try:
   if abs(psutil.Process(previous['pid']).create_time()-previous['create_time'])<1:raise RuntimeError('Coordinator already alive')
  except psutil.NoSuchProcess:pass
 # Also exclude an orphan child after an interrupted coordinator.
 for p in psutil.process_iter(['pid','name','cmdline']):
  if p.pid==os.getpid():continue
  cmd=p.info.get('cmdline') or []
  if any(str(HERE) in a and (a.endswith('data.py') or a.endswith('job.py')) for a in cmd):raise RuntimeError(f'Existing run child{p.pid}; inspect before restarting')
 seal=json.loads((HERE/'launch_seal.json').read_text());assert seal['code']==hashes(),'Unverified source edits after preflight'
 for name in ('optics_check.json','camera_check.json','training_preflight.json'):
  path=HERE/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['checks'][name]
  assert json.loads(path.read_text())['status']=='pass'
 calibration=HERE/'camera_calibration.json';assert hashlib.sha256(calibration.read_bytes()).hexdigest()==seal['camera_calibration_sha256']
 ext=HERE.parent/'canonical_three_phase_universal/models.py';assert hashlib.sha256(ext.read_bytes()).hexdigest()==seal['architecture_sha256']
 started=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=stamp(),workers=workers,python=sys.executable);atomic(HERE/'started.json',started)
 logs=HERE/'logs';logs.mkdir(exist_ok=True);start=clock();wall=time.time()
 stages=[('sampling','data.py',['--workers',str(workers)],HERE/'dataset_manifest.json'),('training','job.py',['train'],HERE/'training/summary.json'),('audit','job.py',['audit'],HERE/'training/joint_result.json')]
 for stage,script,arguments,done in stages:
  if done.exists():continue
  assert hashes()==seal['code'],'Code changed during run; stop before next stage'
  with (logs/f'{stage}.log').open('a',encoding='utf-8') as f:
   child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(HERE/script),*arguments],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
   atomic(HERE/'state.json',dict(status='running',stage=stage,pid=os.getpid(),child_pid=child.pid,updated=stamp(),effective_seconds=clock()-start,wall_seconds=time.time()-wall))
   code=child.wait()
  if code:raise RuntimeError(f'{stage} exited{code}; inspect logs/{stage}.log')
 result=dict(status='complete',assistant_review='pending',updated=stamp(),effective_seconds=clock()-start,wall_seconds=time.time()-wall,notification_configured=False,result='training/joint_result.json')
 atomic(HERE/'state.json',result)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);args=parser.parse_args();awake()
 try:run(args.workers)
 except Exception as error:
  failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),updated=stamp());atomic(HERE/'failure_pipeline.json',failure);atomic(HERE/'state.json',failure);raise
 finally:awake(False)
