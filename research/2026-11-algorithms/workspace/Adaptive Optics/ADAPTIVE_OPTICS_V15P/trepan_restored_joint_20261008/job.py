from common import *
from network import JointNetwork,restore
from objective import Objective
from optics import Optics
from geometry import Geometry
from data import load_dataset
import torch,math,csv,argparse,traceback,psutil
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
EPOCHS=50;BATCH=32
OUT=HERE/'training';OUT.mkdir(exist_ok=True)
def save_torch(path,payload):
 tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def aggregate(w,valid,cohort=None):
 good=valid&np.isfinite(w).all(1);v=w[good]
 result=dict(count=len(w),passes=int(np.sum(good&(w[:,0]<.07))),recovery_rate=float(np.mean(good&(w[:,0]<.07))),invalid=int((~good).sum()),center_mean=float(v[:,0].mean()) if len(v) else None,center_p90=float(np.quantile(v[:,0],.9)) if len(v) else None,nine_field_mean=float(v.mean()) if len(v) else None,both08_rate=float(np.mean(good&(w[:,0]<.08)&(w.mean(1)<.08))))
 if cohort is not None:result['cohorts']={str(c):aggregate(w[cohort==c],valid[cohort==c]) for c in np.unique(cohort)}
 return result
@torch.no_grad()
def evaluate(model,optics,data,ids,save=None):
 x,q,_,cohort,_=data;model.eval();geo=Geometry();ws=[];vs=[];commands=[];supports=[]
 for start in range(0,len(ids),8):
  ix=ids[start:start+8];images=torch.tensor(x[ix].astype(np.float32),device='cuda');command=model.command(images).double();state=torch.tensor(q[ix],dtype=torch.float64,device='cuda')+command
  w,s=optics.metrics(state);wc=w.cpu().numpy();ss=s.cpu().numpy();endpoint=state.cpu().numpy()
  valid=np.array([geo.path(b,a) for a,b in zip(q[ix],endpoint)])&(ss.min(1)>=.98)&np.isfinite(wc).all(1)
  ws.append(wc);vs.append(valid);commands.append(command.cpu().numpy());supports.append(ss)
 w=np.concatenate(ws);valid=np.concatenate(vs);cmd=np.concatenate(commands);support=np.concatenate(supports)
 if save:np.savez_compressed(save,indices=ids,initial=q[ids],command=cmd,endpoint=q[ids]+cmd,wrms=w,valid=valid,support=support,cohort=cohort[ids])
 return aggregate(w,valid,cohort[ids]),(w,valid,cmd)
def curves(history):
 atomic(OUT/'curves.json',dict(epochs=history,metric='whole-assembly noisy sensorless single40D command; center PTTD WRMS<0.07 waves',validation_used_for_selection=True))
 keys=['epoch','train_loss','physical','pose','mechanical','train_rate','validation_rate','center_mean','center_p90','nine_field_mean','invalid','active_seconds']
 with (OUT/'curves.csv.tmp').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
  for h in history:
   writer.writerow({k:h[k] for k in keys if k in h}|dict(train_rate=h['train_probe']['recovery_rate'],validation_rate=h['validation']['recovery_rate'],**{k:h['validation'][k] for k in ('center_mean','center_p90','nine_field_mean','invalid')}))
 os.replace(OUT/'curves.csv.tmp',OUT/'curves.csv')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(11,8));ep=[h['epoch'] for h in history]
 for key in ('train_loss','physical','pose','mechanical'):axs[0,0].plot(ep,[h.get(key,float('nan')) for h in history],label=key)
 for key in ('train_probe','validation'):axs[0,1].plot(ep,[100*h[key]['recovery_rate'] for h in history],label=key)
 axs[0,1].axhline(90,ls=':',c='gray');axs[0,1].set_ylim(0,100);axs[0,1].set_ylabel('Whole-system one-shot recovery (%)')
 for key in ('center_mean','center_p90','nine_field_mean'):axs[1,0].plot(ep,[h['validation'][key] for h in history],label=key)
 axs[1,0].axhline(.07,ls=':',c='gray');axs[1,0].set_ylabel('WRMS (waves)')
 axs[1,1].plot(ep,[h['validation']['invalid'] for h in history],label='Invalid endpoints/command paths')
 for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.25);ax.set_xlabel('Epoch')
 fig.suptitle('Trepan restored NA: joint front/rear50epochs; noisy camera only');fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=140);fig.savefig(OUT/'curves.pdf');plt.close(fig)
def selection(v):return (-v['passes'],v['center_p90'] if v['center_p90'] is not None else 1e9,v['center_mean'] if v['center_mean'] is not None else 1e9)
def reference(data,train_ids):
 total=np.zeros((9,128,128),np.float64)
 for a in range(0,len(train_ids),100):total+=data[0][train_ids[a:a+100]].sum(0,dtype=np.float64)
 return (total/len(train_ids)).astype(np.float32)
def preflight():
 torch.manual_seed(9108);data=load_dataset();assert len(data[0])==100
 ref=reference(data,np.arange(100));model=JointNetwork(ref).cuda();objective=Objective();full=Optics(grid=65,device='cuda')
 x=torch.tensor(data[0][:BATCH].astype(np.float32),device='cuda');q=torch.tensor(data[1][:BATCH],dtype=torch.float64,device='cuda')
 with torch.no_grad():assert model.command(x).count_nonzero()==0
 optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);timings=[];losses=[]
 for i in range(5):
  model.train();start=clock();optimizer.zero_grad(set_to_none=True);command=model.command(x);loss,parts=objective.loss(command,q)
  assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
  torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();torch.cuda.synchronize();timings.append(clock()-start);losses.append(float(loss.detach()))
 for label,module in [('raw',model.node_encoder),('cross_module',model.module_attention),('cnn',model.base.stem)]:
  assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0,label
 # Scoring must count the full40D action and certify the entire actuator path.
 start=clock();metric,_=evaluate(model,full,data,np.r_[np.arange(16),np.arange(50,66)]);eval_seconds=clock()-start
 gridchecks=[];coarse=objective.center;fine=Optics(grid=129,fields=[0],device='cuda')
 with torch.no_grad():
  a=coarse.metrics(q)[0];b=fine.metrics(q)[0];griderror=float(abs(a-b).max())
 assert griderror<.005,griderror
 # Invalid endpoint penalty exceeds every valid optical term even at enormous WFE.
 t=torch.tensor([1.,100.,1e6]);ell=torch.log1p(t*t);assert (ell/(1+ell)<1).all()
 result=dict(status='pass',discarded_probe_updates=5,training_updates=0,losses=losses,batch=BATCH,step_seconds=timings,steady_step_seconds=float(np.median(timings[2:])),evaluation_seconds_per_case=eval_seconds/32,gpu=torch.cuda.get_device_name(),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,full_action_probe=metric,training_score_grid65_vs129_max_error=griderror,parameters=sum(p.numel() for p in model.parameters()),code=hashes())
 atomic(HERE/'training_preflight.json',result);print(json.dumps(result),flush=True)
def train():
 data=load_dataset();x,q,index,cohort,initial=data;assert len(x)==15000
 train_ids=np.where(index<7000)[0];val_ids=np.where(index>=7000)[0];probe=np.random.default_rng(910).choice(train_ids,500,replace=False)
 assert len(train_ids)==14000 and len(val_ids)==1000
 torch.manual_seed(9102026);torch.cuda.manual_seed_all(9102026)
 ref=reference(data,train_ids);model=JointNetwork(ref).cuda();obj=Objective();full=Optics(grid=65,device='cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 history=[];first_epoch=1;first_batch=0;accum=np.zeros(5);processed=0;best=None;epoch_active=0.;start=clock();code=hashes()
 if (OUT/'resume.pt').exists():
  resume=torch.load(OUT/'resume.pt',map_location='cpu',weights_only=False)
  assert resume['code']==code,'Source changed after checkpoint; explicit migration required'
  model.load_state_dict(resume['model']);optimizer.load_state_dict(resume['optimizer']);history=resume['history'];first_epoch=resume['epoch'];first_batch=resume['batch'];accum=np.array(resume['accum']);processed=resume['processed'];best=resume['best'];epoch_active=resume['epoch_active']
  torch.set_rng_state(resume['rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
 def payload():return dict(model=model.state_dict(),reference=ref,code=code,protocol='restored joint40D; sensorless noisy intensity',scales=SCALES.tolist())
 def checkpoint(epoch,batch):
  save_torch(OUT/'resume.pt',payload()|dict(optimizer=optimizer.state_dict(),history=history,epoch=epoch,batch=batch,accum=accum.tolist(),processed=processed,best=best,epoch_active=epoch_active,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
 if not history:
  v,_=evaluate(model,full,data,val_ids);tr,_=evaluate(model,full,data,probe)
  history.append(dict(epoch=0,train_loss=None,physical=None,pose=None,mechanical=None,train_probe=tr,validation=v,active_seconds=0))
  best=selection(v);save_torch(OUT/'candidate.pt',payload()|dict(epoch=0,validation=v));curves(history)
  atomic(OUT/'initial_distribution.json',dict(validation=aggregate(initial[val_ids],np.ones(1000,bool),cohort[val_ids]),train=aggregate(initial[train_ids],np.ones(14000,bool),cohort[train_ids])))
  checkpoint(1,0)
 for epoch in range(first_epoch,EPOCHS+1):
  order=np.random.default_rng(6100800+epoch).permutation(train_ids);total_batches=math.ceil(len(order)/BATCH)
  lr=3e-6+.5*(3e-4-3e-6)*(1+math.cos(math.pi*(epoch-1)/EPOCHS))
  for group in optimizer.param_groups:group['lr']=lr
  for batch in range(first_batch,total_batches):
   tick=clock();ids=order[batch*BATCH:(batch+1)*BATCH];images=torch.tensor(x[ids].astype(np.float32),device='cuda');target=torch.tensor(q[ids],dtype=torch.float64,device='cuda')
   model.train();optimizer.zero_grad(set_to_none=True);command=model.command(images);loss,parts=obj.loss(command,target)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stop without committing update')
   loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
   n=len(ids);accum+=n*np.array([float(loss.detach()),parts['physical'],parts['pose'],parts['mechanical'],parts['wrms']]);processed+=n;epoch_active+=clock()-tick
   if (batch+1)%20==0 or batch+1==total_batches:
    checkpoint(epoch,batch+1);atomic(OUT/'state.json',dict(status='training',epoch=epoch,total_epochs=EPOCHS,batch=batch+1,batches=total_batches,loss=(accum/processed).tolist(),epoch_active_seconds=epoch_active,updated=stamp(),pid=os.getpid(),lr=lr))
  ev=clock();v,_=evaluate(model,full,data,val_ids);tr,_=evaluate(model,full,data,probe);epoch_active+=clock()-ev
  score=selection(v)
  if best is None or score<tuple(best):best=score;save_torch(OUT/'candidate.pt',payload()|dict(epoch=epoch,validation=v))
  averages=accum/processed;history.append(dict(epoch=epoch,train_loss=float(averages[0]),physical=float(averages[1]),pose=float(averages[2]),mechanical=float(averages[3]),train_probe=tr,validation=v,active_seconds=epoch_active,lr=lr))
  curves(history);save_torch(OUT/'last.pt',payload()|dict(epoch=epoch,validation=v));print(json.dumps(history[-1]),flush=True)
  first_batch=0;accum=np.zeros(5);processed=0;epoch_active=0.;checkpoint(epoch+1,0)
 atomic(OUT/'summary.json',dict(status='training_complete',epochs=50,assistant_review='pending',selected=torch.load(OUT/'candidate.pt',map_location='cpu',weights_only=False)['epoch'],updated=stamp()))
def audit():
 from check_optics import scalar_reference
 from optics import numpy_trace,references
 data=load_dataset();ids=np.where(data[2]>=7000)[0];model,payload=restore(OUT/'candidate.pt');full=Optics(grid=65,device='cuda');metric,arrays=evaluate(model,full,data,ids,OUT/'joint_validation1000.npz')
 # Independent full85-surface NumPy replay (no cached prefix) on24fixed cases.
 geo=Geometry();xy=disk(65);A=np.c_[np.ones(len(xy)),xy,(xy*xy).sum(1)];ref=references();rows=[];scalar=[]
 endpoints=data[1][ids]+arrays[2]
 for k in np.linspace(0,len(ids)-1,24,dtype=int):
  for fi in (0,2,4):
   pp,dd,op,valid=numpy_trace(xy,FIELDS[fi],endpoints[k]);delta=pp-ref[fi];b=(delta*dd).sum(1);disc=b*b-(delta*delta).sum(1)+10000;valid&=disc>0
   wave=(op+SPEC['n_image']*(-b-np.sqrt(np.maximum(disc,0))))/.000910
   res=wave[valid]-A[valid]@np.linalg.lstsq(A[valid],wave[valid],rcond=None)[0];w=float(np.sqrt(np.mean(res**2)));error=abs(w-arrays[0][k,fi]);rows.append(dict(case=int(k),field=fi,error=error,reference=w))
   pp0,dd0,op0=scalar_reference(endpoints[k],xy[len(xy)//2],FIELDS[fi]);j=len(xy)//2;scalar.append(max(np.max(abs(pp0-pp[j])),np.max(abs(dd0-dd[j])),abs(op0-op[j])))
 assert max(r['error'] for r in rows)<2e-6 and max(scalar)<2e-8
 # Denser center-only score for every validation case; never hides disagreements.
 dense=Optics(grid=129,fields=[0],device='cuda');wv=[]
 with torch.no_grad():
  for k in range(0,len(ids),8):wv.extend(dense.metrics(torch.tensor(endpoints[k:k+8],dtype=torch.float64,device='cuda'))[0][:,0].cpu().tolist())
 wv=np.asarray(wv);np.savez_compressed(OUT/'center129_values.npz',indices=ids,wrms=wv,valid=arrays[1])
 result=dict(status='computed_independently_checked',assistant_review='pending',checkpoint_epoch=payload['epoch'],validation65=metric,initial_validation=aggregate(data[4][ids],np.ones(len(ids),bool),data[3][ids]),center129_passes=int(np.sum((wv<.07)&arrays[1])),center_grid_max_error=float(np.max(abs(wv-arrays[0][:,0]))),center_grid_threshold_disagreements=int(np.sum((wv<.07)!=(arrays[0][:,0]<.07))),independent_checks=rows,scalar_rayoptics_max_error=max(scalar),command='one noisy acquisition, one joint40D prediction; no oracle clipping/gain, no feedback',validation_is_selection_set=True,updated=stamp())
 atomic(OUT/'joint_result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','train','audit']);args=parser.parse_args();awake()
 try:globals()[args.stage]()
 except Exception as error:
  atomic(HERE/f'failure_{args.stage}.json',dict(error=repr(error),traceback=traceback.format_exc(),updated=stamp()));raise
 finally:awake(False)
