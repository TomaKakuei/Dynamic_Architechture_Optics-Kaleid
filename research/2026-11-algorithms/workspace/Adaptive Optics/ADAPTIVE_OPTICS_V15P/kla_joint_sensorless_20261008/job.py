from common import *
from network import JointNetwork,restore
from objective import Objective
from optics import Optics
from geometry import Geometry
from data import load_dataset
import torch,math,csv,argparse,traceback
torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
EPOCHS=50;BATCH=32
OUT=HERE/'training';OUT.mkdir(exist_ok=True)
def save_torch(path,payload):
 tmp=path.with_suffix('.tmp');torch.save(payload,tmp);os.replace(tmp,path)
def aggregate(fields,valid,cohort=None,wrms=None):
 fields=np.asarray(fields);scores=fields.mean(1)-.3*fields.std(1);good=np.asarray(valid)&np.isfinite(scores);v=scores[good]
 result=dict(count=len(scores),passes=int(np.sum(good&(scores>=.45))),recovery_rate=float(np.mean(good&(scores>=.45))),invalid=int((~good).sum()),q_mean=float(v.mean()) if len(v) else None,q_p10=float(np.quantile(v,.1)) if len(v) else None,field_mtf_mean=fields[good].mean(0).tolist() if len(v) else None)
 if wrms is not None:result['wrms_mean']=float(np.asarray(wrms)[good].mean()) if len(v) else None
 if cohort is not None:result['cohorts']={str(c+2):aggregate(fields[cohort==c],good[cohort==c],wrms=None if wrms is None else wrms[cohort==c]) for c in np.unique(cohort)}
 return result
@torch.no_grad()
def evaluate(model,optics,data,ids,save=None):
 x,q,_,cohort,_=data;model.eval();geo=Geometry();fields=[];valids=[];commands=[];supports=[];wrms=[]
 for k in range(0,len(ids),8):
  ix=ids[k:k+8];images=torch.tensor(x[ix].astype(np.float32),device='cuda');cmd=model.command(images).double();state=torch.tensor(q[ix],dtype=torch.float64,device='cuda')+cmd
  result=optics.metrics(state);ep=state.cpu().numpy();good=np.array([geo.path(b,a) for a,b in zip(q[ix],ep)])&result['valid'].cpu().numpy()
  fields.append(result['fields'].cpu().numpy());valids.append(good);commands.append(cmd.cpu().numpy());supports.append(result['support'].cpu().numpy());wrms.append(result['wrms'].cpu().numpy())
 f=np.concatenate(fields);v=np.concatenate(valids);cmd=np.concatenate(commands);s=np.concatenate(supports);w=np.concatenate(wrms)
 if save:np.savez_compressed(save,indices=ids,initial=q[ids],command=cmd,endpoint=q[ids]+cmd,fields=f,score=f.mean(1)-.3*f.std(1),wrms=w,valid=v,support=s,cohort=cohort[ids])
 return aggregate(f,v,cohort[ids],w),(f,v,cmd,w)
def curves(history):
 atomic(OUT/'curves.json',dict(epochs=history,metric='Joint40D one-shot Q2000 >= 0.45, mean tangential9fieldMTF minus0.3 population SD',validation_used_for_selection=True))
 keys=['epoch','train_loss','physical','pose','mechanical','train_rate','validation_rate','q_mean','q_p10','wrms_mean','invalid','active_seconds']
 with (OUT/'curves.csv.tmp').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
  for h in history:writer.writerow({k:h[k] for k in keys if k in h}|dict(train_rate=h['train_probe']['recovery_rate'],validation_rate=h['validation']['recovery_rate'],**{k:h['validation'][k] for k in ('q_mean','q_p10','wrms_mean','invalid')}))
 os.replace(OUT/'curves.csv.tmp',OUT/'curves.csv')
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(11,8));ep=[h['epoch'] for h in history]
 for k in ('train_loss','physical','pose','mechanical'):axs[0,0].plot(ep,[h.get(k,float('nan')) for h in history],label=k)
 for k in ('train_probe','validation'):axs[0,1].plot(ep,[100*h[k]['recovery_rate'] for h in history],label=k)
 axs[0,1].set_ylim(0,100);axs[0,1].set_ylabel('Whole-system one-shot recovery (%)')
 for k in ('q_mean','q_p10'):axs[1,0].plot(ep,[h['validation'][k] for h in history],label=k)
 axs[1,0].axhline(.45,ls=':',c='gray');axs[1,0].set_ylabel('Q2000')
 axs[1,1].plot(ep,[h['validation']['invalid'] for h in history],label='Uncertified paths/invalid optics')
 for ax in axs.flat:ax.legend(fontsize=8);ax.grid(alpha=.25);ax.set_xlabel('Epoch')
 fig.suptitle(f'KLA: joint modules2/3/4, noisy camera, completed {ep[-1]} epochs');fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=140);fig.savefig(OUT/'curves.pdf');plt.close(fig)
def selection(v):return (-v['passes'],-v['q_p10'] if v['q_p10'] is not None else 1e9,-v['q_mean'] if v['q_mean'] is not None else 1e9)
def reference(data,ids):
 total=np.zeros((9,128,128),np.float64)
 for k in range(0,len(ids),100):total+=data[0][ids[k:k+100]].sum(0,dtype=np.float64)
 return (total/len(ids)).astype(np.float32)
def preflight():
 data=load_dataset();assert len(data[0])==150;torch.manual_seed(1933);ref=reference(data,np.arange(150));model=JointNetwork(ref).cuda();obj=Objective();full=Optics(grid=65,device='cuda')
 ids=np.r_[np.arange(11),np.arange(50,61),np.arange(100,110)];x=torch.tensor(data[0][ids].astype(np.float32),device='cuda');q=torch.tensor(data[1][ids],dtype=torch.float64,device='cuda')
 with torch.no_grad():assert model.command(x).count_nonzero()==0
 optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);timings=[];losses=[]
 for _ in range(6):
  model.train();start=clock();optimizer.zero_grad(set_to_none=True);loss,parts=obj.loss(model.command(x),q);assert torch.isfinite(loss);loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize();timings.append(clock()-start);losses.append(float(loss.detach()))
 for label,module in [('raw',model.node_encoder),('cross_module',model.module_attention),('cnn',model.base.stem)]:assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)>0,label
 start=clock();metric,_=evaluate(model,full,data,ids);ev=(clock()-start)/len(ids)
 with torch.no_grad():
  inv,parts=obj.loss(-q.float(),q);assert parts['mechanical']==0
  a=obj.optics.metrics(q)['wrms'];b=full.metrics(q)['wrms'];relative=float((abs(a-b)/b.clamp_min(.01)).max())
 assert relative<.04,relative
 # Batch-size drift checked with TF32 disabled, under the same eval model.
 model.eval()
 with torch.no_grad():a=model.command(x);b=torch.cat([model.command(x[i:i+8]) for i in range(0,len(x),8)])
 drift=float(abs(a-b).max());assert drift<2e-5,drift
 result=dict(status='pass',discarded_probe_updates=6,training_updates=0,batch=BATCH,losses=losses,step_seconds=timings,steady_step_seconds=float(np.median(timings[2:])),evaluation_seconds_per_case=ev,gpu=torch.cuda.get_device_name(),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,probe=metric,wrms_grid33_vs65_max_relative_error=relative,batch_inference_max_difference=drift,parameters=sum(p.numel() for p in model.parameters()),code=hashes())
 atomic(HERE/'training_preflight.json',result);print(json.dumps(result),flush=True)
def train(until_epoch=20):
 assert 1<=until_epoch<=EPOCHS,'Requested stop must be within the registered50epoch schedule'
 data=load_dataset();x,q,index,cohort,initial=data;assert len(x)==TOTAL;train_ids=np.where(index<TRAIN)[0];val_ids=np.where(index>=TRAIN)[0]
 assert len(train_ids)==3*TRAIN and len(val_ids)==3*VALIDATION
 probe=np.random.default_rng(1933).choice(train_ids,500,replace=False);torch.manual_seed(19332026);torch.cuda.manual_seed_all(19332026)
 ref=reference(data,train_ids);model=JointNetwork(ref).cuda();obj=Objective();full=Optics(grid=65,device='cuda');optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
 history=[];first_epoch=1;first_batch=0;accum=np.zeros(5);processed=0;best=None;epoch_active=0.;code=hashes()
 if (OUT/'resume.pt').exists():
  r=torch.load(OUT/'resume.pt',map_location='cpu',weights_only=False);assert r['code']==code,'Source changed; explicit migration required'
  model.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer']);history=r['history'];first_epoch=r['epoch'];first_batch=r['batch'];accum=np.array(r['accum']);processed=r['processed'];best=r['best'];epoch_active=r['epoch_active'];torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
 assert first_epoch<=until_epoch or (first_epoch==until_epoch+1 and first_batch==0),'Checkpoint already exceeds requested training budget'
 def payload():return dict(model=model.state_dict(),reference=ref,code=code,protocol='KLA noisy camera9fields joint40D; no oracle clipping',scales=SCALES.tolist())
 def checkpoint(epoch,batch):save_torch(OUT/'resume.pt',payload()|dict(optimizer=optimizer.state_dict(),history=history,epoch=epoch,batch=batch,accum=accum.tolist(),processed=processed,best=best,epoch_active=epoch_active,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
 if not history:
  v,_=evaluate(model,full,data,val_ids);tr,_=evaluate(model,full,data,probe);history.append(dict(epoch=0,train_loss=None,physical=None,pose=None,mechanical=None,train_probe=tr,validation=v,active_seconds=0))
  best=selection(v);save_torch(OUT/'candidate.pt',payload()|dict(epoch=0,validation=v));curves(history);atomic(OUT/'initial_distribution.json',dict(validation=aggregate(initial[val_ids],np.ones(len(val_ids),bool),cohort[val_ids]),train=aggregate(initial[train_ids],np.ones(len(train_ids),bool),cohort[train_ids])));checkpoint(1,0)
 # The stopping budget does not change the original learning-rate schedule.
 for epoch in range(first_epoch,until_epoch+1):
  order=np.random.default_rng(19331000+epoch).permutation(train_ids);batches=math.ceil(len(order)/BATCH);lr=3e-6+.5*(3e-4-3e-6)*(1+math.cos(math.pi*(epoch-1)/EPOCHS))
  for g in optimizer.param_groups:g['lr']=lr
  for batch in range(first_batch,batches):
   tick=clock();ids=order[batch*BATCH:(batch+1)*BATCH];images=torch.tensor(x[ids].astype(np.float32),device='cuda');target=torch.tensor(q[ids],dtype=torch.float64,device='cuda')
   model.train();optimizer.zero_grad(set_to_none=True);loss,parts=obj.loss(model.command(images),target)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss; stop before update')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
   n=len(ids);accum+=n*np.array([float(loss.detach()),parts['physical'],parts['pose'],parts['mechanical'],parts['wrms']]);processed+=n;epoch_active+=clock()-tick
   if (batch+1)%20==0 or batch+1==batches:
    checkpoint(epoch,batch+1);atomic(OUT/'state.json',dict(status='training',epoch=epoch,total_epochs=until_epoch,schedule_epochs=EPOCHS,batch=batch+1,batches=batches,loss=(accum/processed).tolist(),epoch_active_seconds=epoch_active,updated=stamp(),pid=os.getpid(),lr=lr))
  tick=clock();v,_=evaluate(model,full,data,val_ids);tr,_=evaluate(model,full,data,probe);epoch_active+=clock()-tick;score=selection(v)
  if best is None or score<tuple(best):best=score;save_torch(OUT/'candidate.pt',payload()|dict(epoch=epoch,validation=v))
  avg=accum/processed;history.append(dict(epoch=epoch,train_loss=float(avg[0]),physical=float(avg[1]),pose=float(avg[2]),mechanical=float(avg[3]),train_probe=tr,validation=v,active_seconds=epoch_active,lr=lr));curves(history);save_torch(OUT/'last.pt',payload()|dict(epoch=epoch,validation=v));print(json.dumps(history[-1]),flush=True)
  first_batch=0;accum=np.zeros(5);processed=0;epoch_active=0.;checkpoint(epoch+1,0)
 assert history[-1]['epoch']==until_epoch
 atomic(OUT/'summary.json',dict(status='training_complete',epochs=until_epoch,schedule_epochs=EPOCHS,continuation_available=until_epoch<EPOCHS,next_epoch=until_epoch+1,assistant_review='pending',selected=torch.load(OUT/'candidate.pt',map_location='cpu',weights_only=False)['epoch'],updated=stamp()))
 atomic(OUT/'state.json',dict(status='training_complete',epoch=until_epoch,total_epochs=until_epoch,schedule_epochs=EPOCHS,continuation_available=until_epoch<EPOCHS,updated=stamp()))
def audit():
 from check_optics import numpy_metric,scalar_reference
 from optics import numpy_trace
 data=load_dataset();ids=np.where(data[2]>=TRAIN)[0];model,payload=restore(OUT/'candidate.pt');full=Optics(grid=65,device='cuda');metric,arr=evaluate(model,full,data,ids,OUT/'joint_validation3000.npz');endpoints=data[1][ids]+arr[2];rows=[];rayerrors=[]
 for k in np.linspace(0,len(ids)-1,24,dtype=int):
  ref=numpy_metric(endpoints[k]);score=float(arr[0][k].mean()-.3*arr[0][k].std());rows.append(dict(case=int(k),score_error=abs(score-ref['score'])))
  for fi in (0,4,8):
   xy=np.array([.31,-.57]);p,d,op,_=numpy_trace(xy[None],SPEC['angles_rad'][fi],endpoints[k]);pp,dd,oo=scalar_reference(endpoints[k],xy,fi);rayerrors.append(max(abs(pp-p[0]).max(),abs(dd-d[0]).max(),abs(oo-op[0])))
 assert max(r['score_error'] for r in rows)<1e-7 and max(rayerrors)<2e-8
 fine=Optics(grid=129,device='cuda');fields=[];supports=[]
 with torch.no_grad():
  for k in range(0,len(ids),4):
   m=fine.metrics(torch.tensor(endpoints[k:k+4],dtype=torch.float64,device='cuda'));fields.append(m['fields'].cpu().numpy());supports.append(m['support'].cpu().numpy())
 fields=np.concatenate(fields);supports=np.concatenate(supports);valid=arr[1]&(supports.min(1)>=.98);scores=fields.mean(1)-.3*fields.std(1);coarse=arr[0].mean(1)-.3*arr[0].std(1)
 np.savez_compressed(OUT/'grid129_values.npz',indices=ids,fields=fields,score=scores,valid=valid,support=supports)
 # Flag every threshold disagreement, and resolve borderline scores at257.
 selected=np.where((abs(scores-.45)<.02)|((scores>=.45)!=(coarse>=.45)))[0];dense=Optics(grid=257,device='cuda');dense_rows=[]
 with torch.no_grad():
  for k in selected:
   m=dense.metrics(torch.tensor(endpoints[k:k+1],dtype=torch.float64,device='cuda'));dense_rows.append(dict(case=int(k),q257=float(m['score'].item()),support=float(m['support'].min().item())))
 result=dict(status='computed_independently_checked',assistant_review='pending',checkpoint_epoch=payload['epoch'],validation65=metric,initial_validation=aggregate(data[4][ids],np.ones(len(ids),bool),data[3][ids]),validation129=aggregate(fields,valid,data[3][ids]),grid65_vs129_max_score_error=float(abs(scores-coarse).max()),grid_threshold_disagreements=int(np.sum((scores>=.45)!=(coarse>=.45))),grid257_borderline=dense_rows,independent_numpy_checks=rows,scalar_rayoptics_max_error=float(max(rayerrors)),command='one noisy9field acquisition, joint40D one-shot; no ideal input/oracle clipping/gain/feedback',validation_is_selection_set=True,updated=stamp())
 result['training_epochs']=json.loads((OUT/'summary.json').read_text())['epochs'];result['candidate_sha256']=hashlib.sha256((OUT/'candidate.pt').read_bytes()).hexdigest()
 atomic(OUT/'joint_result.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preflight','train','audit']);parser.add_argument('--until-epoch',type=int,default=20);args=parser.parse_args();awake()
 try:
  if args.stage=='train':train(args.until_epoch)
  else:globals()[args.stage]()
 except Exception as error:atomic(HERE/f'failure_{args.stage}.json',dict(error=repr(error),traceback=traceback.format_exc(),updated=stamp()));raise
 finally:awake(False)
