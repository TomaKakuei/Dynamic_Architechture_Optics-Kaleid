from camera import Camera,WaveFit
from optics import Optics
from sampling import sample
from common import *
import torch
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
WORKER=None
def init_worker():
 global WORKER
 torch.set_num_threads(1);WORKER=(WaveFit(),Optics(grid=65))
def produce(task):
 cohort,index=task;fit,metric=WORKER
 q,rejections=sample(cohort,index)
 # A failed fit/domain check stops the run; do not change optical difficulty
 # by silently resampling a sample that the image renderer cannot represent.
 reps=fit.fit(q)
 with torch.no_grad():w,s=metric.metrics(torch.tensor(q[None],dtype=torch.float64))
 return dict(cohort=cohort,index=index,q=q,reps=reps,initial_wrms=w[0].numpy(),support=s[0].numpy(),mechanical_rejections=rejections)
def save_block(path,rows,images):
 temp=path.with_suffix('.tmp')
 with temp.open('wb') as f:np.savez_compressed(f,speckles=np.stack(images),q=np.stack([r['q'] for r in rows]),initial_wrms=np.stack([r['initial_wrms'] for r in rows]),support=np.stack([r['support'] for r in rows]),indices=np.array([r['index'] for r in rows]),cohort=np.array([r['cohort'] for r in rows]),mechanical_rejections=np.array([r['mechanical_rejections'] for r in rows]),fit_error=np.array([max(x['independent_fit_rms_waves'] for x in r['reps']) for r in rows]))
 os.replace(temp,path)
def generate(workers=6,pilot=False):
 out=HERE/'data';out.mkdir(exist_ok=True);cal=json.loads((HERE/'camera_calibration.json').read_text());camera=Camera(calibration=cal);grid=cal['grid'];start=clock();new=0
 count=50 if pilot else 7500;block=50
 with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
  # At most2blocks in flight: bounded RAM, deterministic per-sample seeds,
  # CPU ray/phase work overlaps GPU integration within this one generator.
  jobs=[]
  for c,b in [(c,b) for b in range(count//block) for c in (0,1)]:
   path=out/f'cohort{c}_{b:03d}.npz'
   if path.exists():
    with np.load(path) as z:
     assert z['speckles'].shape==(block,9,128,128) and z['speckles'].dtype==np.uint16
     assert z['q'].shape==(block,40) and np.isfinite(z['q']).all()
     assert np.array_equal(z['indices'],np.arange(b*block,(b+1)*block)) and (z['cohort']==c).all()
   else:jobs.append((c,b))
  pending=[]
  for c,b in jobs:
   path=out/f'cohort{c}_{b:03d}.npz'
   pending.append((path,[pool.submit(produce,(c,b*block+j)) for j in range(block)]))
   if len(pending)<2 and (c,b)!=jobs[-1]:continue
   while pending:
    dest,futures=pending.pop(0);rows=[];images=[];bt=clock()
    for f in futures:
     row=f.result();seed=91000000+row['cohort']*100000+row['index'];images.append(camera.capture(row['reps'],seed,grid));rows.append(row)
    save_block(dest,rows,images);new+=len(rows)
    completed=len(list(out.glob('cohort*.npz')))*block
    result=dict(status='sampling',completed=completed,total=15000,new_this_session=new,active_seconds=clock()-start,seconds_per_new_sample=(clock()-start)/new,block_seconds=clock()-bt,last_file=dest.name,updated=stamp(),initial_passes_in_block=int(sum(r['initial_wrms'][0]<.07 for r in rows)))
    atomic(HERE/'generation_progress.json',result);print(json.dumps(result),flush=True)
 # CUDA caches and pool processes are released before training starts.
 if not pilot:
  files=sorted(out.glob('cohort*.npz'));assert len(files)==300
  manifest=dict(files=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],per_cohort=dict(train=7000,validation=500),total=15000,whole_assembly_observations=True,training_rows=14000,validation_rows=1000,updated=stamp())
  atomic(HERE/'dataset_manifest.json',manifest)
def load_dataset():
 files=sorted((HERE/'data').glob('cohort*.npz'));xs=[];qs=[];ids=[];cohorts=[];initial=[]
 for p in files:
  with np.load(p) as z:
   xs.append(z['speckles']);qs.append(z['q']);ids.append(z['indices']);cohorts.append(z['cohort']);initial.append(z['initial_wrms'])
 return np.concatenate(xs),np.concatenate(qs),np.concatenate(ids),np.concatenate(cohorts),np.concatenate(initial)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
 awake()
 try:generate(args.workers,args.pilot)
 finally:awake(False)
