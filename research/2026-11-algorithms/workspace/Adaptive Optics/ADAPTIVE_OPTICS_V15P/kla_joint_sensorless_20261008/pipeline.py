"""Restart-safe generator -> budgeted KLA joint training -> independent audit."""
from common import *
import sys,subprocess,traceback,psutil,argparse,shutil
def stage_complete(stage,done,until_epoch):
 if not done.exists():return False
 if stage=='sampling':return True
 record=json.loads(done.read_text())
 return record.get('epochs' if stage=='training' else 'training_epochs',0)>=until_epoch

def run(workers,until_epoch=20):
 assert 1<=until_epoch<=50,'Requested stop must be within the registered50epoch schedule'
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
 for name in seal['checks']:
  path=HERE/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==seal['checks'][name]
  assert json.loads(path.read_text())['status']=='pass'
 calibration=HERE/'camera_calibration.json';assert hashlib.sha256(calibration.read_bytes()).hexdigest()==seal['camera_calibration_sha256']
 ext=HERE.parent/'canonical_three_phase_universal/models.py';assert hashlib.sha256(ext.read_bytes()).hexdigest()==seal['architecture_sha256']
 started=dict(pid=os.getpid(),create_time=psutil.Process().create_time(),utc=stamp(),workers=workers,python=sys.executable,until_epoch=until_epoch);atomic(HERE/'started.json',started)
 logs=HERE/'logs';logs.mkdir(exist_ok=True);start=clock();wall=time.time()
 stages=[('sampling','data.py',['--workers',str(workers)],HERE/'dataset_manifest.json'),('training','job.py',['train','--until-epoch',str(until_epoch)],HERE/'training/summary.json'),('audit','job.py',['audit'],HERE/'training/joint_result.json')]
 for stage,script,arguments,done in stages:
  if stage_complete(stage,done,until_epoch):continue
  if stage=='training' and done.exists():
   # Preserve the previous budget's complete result before an authorized extension.
   prior=json.loads(done.read_text());archive=HERE/'training/budget_archives'/f"epochs_{prior['epochs']:03d}_{hashlib.sha256(done.read_bytes()).hexdigest()[:12]}"
   if not (archive/'archive_complete.json').exists():
    archive.mkdir(parents=True,exist_ok=True)
    for saved in (HERE/'training').iterdir():
     if saved.is_file() and saved.suffix!='.tmp':shutil.copy2(saved,archive/saved.name)
    atomic(archive/'archive_complete.json',dict(epochs=prior['epochs'],updated=stamp()))
  assert hashes()==seal['code'],'Code changed during run; stop before next stage'
  with (logs/f'{stage}.log').open('a',encoding='utf-8') as f:
   child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(HERE/script),*arguments],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
   atomic(HERE/'state.json',dict(status='running',stage=stage,pid=os.getpid(),child_pid=child.pid,until_epoch=until_epoch,updated=stamp(),effective_seconds=clock()-start,wall_seconds=time.time()-wall))
   code=child.wait()
  if code:raise RuntimeError(f'{stage} exited{code}; inspect logs/{stage}.log')
 result=dict(status='complete',training_epochs=json.loads((HERE/'training/summary.json').read_text())['epochs'],requested_epochs=until_epoch,continuation_available=until_epoch<50,assistant_review='pending',updated=stamp(),effective_seconds=clock()-start,wall_seconds=time.time()-wall,notification_configured=False,result='training/joint_result.json')
 atomic(HERE/'state.json',result)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--until-epoch',type=int,default=20);args=parser.parse_args();awake()
 try:run(args.workers,args.until_epoch)
 except Exception as error:
  failure=dict(status='failed',error=repr(error),traceback=traceback.format_exc(),updated=stamp());atomic(HERE/'failure_pipeline.json',failure);atomic(HERE/'state.json',failure);raise
 finally:awake(False)
