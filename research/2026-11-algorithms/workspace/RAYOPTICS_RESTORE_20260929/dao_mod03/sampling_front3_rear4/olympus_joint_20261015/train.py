from common import *
from network import initialize,restore,JointNetwork
from objective import Objective
from data import Blocks
import numpy as np
import torch, math, gc, csv
from concurrent.futures import ThreadPoolExecutor

OUT=HERE/'training'
OUT.mkdir(exist_ok=True)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True

def save(path,value):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    commit(temporary,path)

def aggregate(w,v):
    good=v&np.isfinite(w).all(1)
    x=w[good]
    return dict(count=len(w),passes=int(np.sum(good&(w[:,0]<.07))),rate=float(np.mean(good&(w[:,0]<.07))),
        invalid=int((~good).sum()),center_mean=float(x[:,0].mean()) if len(x) else None,
        center_p90=float(np.quantile(x[:,0],.9)) if len(x) else None,
        field_mean=float(x.mean()) if len(x) else None,
        mean_scope='finite physically valid cases; invalids remain failures in pass-rate denominator')

def score(v):
    return (-v['passes'],v['center_p90'] if v['center_p90'] is not None else 1e9,
             v['center_mean'] if v['center_mean'] is not None else 1e9)

@torch.no_grad()
def evaluate(model,obj,blocks,full=False):
    batches=range(280,300) if full else (280,285,290,295)
    ws=[];vs=[];commands=[];qs=[];ids=[]
    for b in batches:
        x,q=blocks.read('joint',b)
        for start in range(0,25,5):
            xx=torch.as_tensor(x[start:start+5].astype(np.float32),device='cuda')
            qq=torch.as_tensor(q[start:start+5],dtype=torch.float64,device='cuda')
            w,v,c=obj.metrics(model,xx,qq)
            ws.append(w);vs.append(v);commands.append(c);qs.append(q[start:start+5])
        ids.extend(range(b*25,(b+1)*25))
    w=np.concatenate(ws);v=np.concatenate(vs)
    return aggregate(w,v),dict(index=np.array(ids),initial=np.concatenate(qs),command=np.concatenate(commands),wrms=w,valid=v)

def curves(history):
    atomic(OUT/'curves.json',dict(epochs=history,validation_used_for_selection=True,independent_review=False))
    keys=['epoch','loss','validation_passes','validation_count','center_mean','center_p90','invalid','active_seconds']
    with (OUT/'curves.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for h in history:
            v=h['validation']
            writer.writerow(dict(epoch=h['epoch'],loss=h.get('loss'),validation_passes=v['passes'],
                validation_count=v['count'],center_mean=v['center_mean'],center_p90=v['center_p90'],
                invalid=v['invalid'],active_seconds=h['active_seconds']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.7))
    e=[h['epoch'] for h in history]
    axes[0].plot(e,[h.get('loss',np.nan) for h in history]);axes[0].set_title('Joint training loss')
    axes[1].plot(e,[100*h['validation']['rate'] for h in history],'.-')
    axes[1].set_title('Whole-system one-shot < 0.07 lambda (%)');axes[1].set_ylim(0,100)
    axes[2].plot(e,[h['validation']['center_mean'] for h in history],label='Mean')
    axes[2].plot(e,[h['validation']['center_p90'] for h in history],label='P90')
    axes[2].axhline(.07,color='gray',ls=':');axes[2].legend();axes[2].set_title('Center PTTD WRMS (lambda)')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.suptitle('Olympus Mod. | joint seven-body correction | noisy full-system image only')
    fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=130);fig.savefig(OUT/'curves.pdf');plt.close(fig)

def startup():
    """One retained optimizer step on actual training samples; no replay audit."""
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model,cal=initialize();model=model.cuda()
    obj=Objective();blocks=Blocks()
    x,q=blocks.read('joint',0)
    start=clock()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    model.train();optimizer.zero_grad(set_to_none=True)
    command=model.command(torch.tensor(x[:4].astype(np.float32),device='cuda'))
    loss,parts=obj.loss(command,torch.tensor(q[:4],device='cuda'))
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite startup training loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    save(OUT/'warm_start.pt',dict(model=model.state_dict(),calibrations=cal,startup_updates=1,
                                 initialization='previous front/rear candidates plus joint attention'))
    result=dict(status='training_startup_complete',retained_optimizer_updates=1,batch=4,loss=float(loss.detach()),
                loss_parts=parts,seconds=clock()-start,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                gpu=torch.cuda.get_device_name(),independent_review=False)
    atomic(HERE/'startup.json',result);print(json.dumps(result),flush=True)

def train():
    blocks=Blocks(capacity=3)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    if (OUT/'resume.pt').exists():
        p=torch.load(OUT/'resume.pt',map_location='cpu',weights_only=False)
        model=JointNetwork(p['calibrations']);model.load_state_dict(p['model']);cal=p['calibrations']
    elif (OUT/'warm_start.pt').exists():
        model,p=restore(OUT/'warm_start.pt',device='cpu');cal=p['calibrations']
    else:
        model,cal=initialize()
    model=model.cuda();obj=Objective()
    old=[];added=[]
    for name,p in model.named_parameters():
        (old if name.startswith(('front.','rear.')) else added).append(p)
    optimizer=torch.optim.AdamW([dict(params=old,lr=2e-5,initial_lr=2e-5),dict(params=added,lr=1e-4,initial_lr=1e-4)],weight_decay=1e-4)
    epoch=1;first=0;history=[];best=None;loss_sum=0.;seen=0;active=0.;micro=25
    if (OUT/'resume.pt').exists():
        r=torch.load(OUT/'resume.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(r['optimizer']);epoch=r['epoch'];first=r['block'];history=r['history']
        best=r['best'];loss_sum=r['loss_sum'];seen=r['seen'];active=r['active'];micro=r.get('micro',25)
        torch.set_rng_state(r['rng']);torch.cuda.set_rng_state_all(r['cuda_rng'])
    def payload():
        return dict(model=model.state_dict(),calibrations=cal,scope='single whole-system noisy camera packet -> joint35D command',
                    acquisition_sets=1,independent_review=False,epochs_budget=EPOCHS)
    def checkpoint(ep,block):
        save(OUT/'resume.pt',payload()|dict(optimizer=optimizer.state_dict(),epoch=ep,block=block,history=history,
             best=best,loss_sum=loss_sum,seen=seen,active=active,micro=micro,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    if not history:
        v,a=evaluate(model,obj,blocks,full=True)
        best=score(v);history=[dict(epoch=0,validation=v,full_validation=True,active_seconds=0)]
        save(OUT/'candidate.pt',payload()|dict(epoch=0,validation=v))
        np.savez_compressed(OUT/'selected_validation.npz',**a)
        curves(history);checkpoint(1,0)
    for ep in range(epoch,EPOCHS+1):
        for g in optimizer.param_groups:
            g['lr']=2e-6+.5*(g['initial_lr']-2e-6)*(1+math.cos(math.pi*(ep-1)/EPOCHS))
        branch='front' if ep%2 else 'rear'
        order=np.random.default_rng(SEED+ep).permutation(280)
        # Each epoch reuses 7000 prior images plus 7000 genuinely joint images.
        schedule=[pair for b in order for pair in (('joint',int(b)),(branch,int(b)))]
        for block in range(first,len(schedule)):
            tick=clock();kind,b=schedule[block];x,q=blocks.read(kind,b)
            model.train()
            retry=True
            while retry:
                retry=False;optimizer.zero_grad(set_to_none=True);total_loss=0.
                try:
                    for s in range(0,len(x),micro):
                        xx=torch.as_tensor(x[s:s+micro].astype(np.float32),device='cuda')
                        qq=torch.as_tensor(q[s:s+micro],dtype=torch.float64,device='cuda')
                        loss,parts=obj.loss(model.command(xx),qq)
                        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite training loss at epoch{ep} block{block}')
                        (loss*(len(xx)/len(x))).backward()
                        total_loss+=float(loss.detach())*len(xx)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    optimizer.step()
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    if micro<=2:raise
                    micro=max(2,micro//2);loss=None;xx=None;qq=None;parts=None;gc.collect();torch.cuda.empty_cache();retry=True
            loss_sum+=total_loss;seen+=len(x);active+=clock()-tick
            if (block+1)%10==0 or block+1==len(schedule):
                checkpoint(ep,block+1)
                state('training',epoch=ep,epochs=EPOCHS,block=block+1,blocks=len(schedule),loss=loss_sum/seen,
                      microbatch=micro,epoch_active_seconds=active,seconds_per_block=active/(block+1),
                      approximate_remaining_train_seconds=((EPOCHS-ep)*len(schedule)+len(schedule)-block-1)*active/(block+1))
        start=clock();full=(ep%5==0 or ep==EPOCHS)
        v,a=evaluate(model,obj,blocks,full=full);active+=clock()-start
        if full and score(v)<tuple(best):
            best=score(v);save(OUT/'candidate.pt',payload()|dict(epoch=ep,validation=v))
            np.savez_compressed(OUT/'selected_validation.npz',**a)
        history.append(dict(epoch=ep,loss=loss_sum/seen,validation=v,full_validation=full,active_seconds=active))
        print('EPOCH',json.dumps(history[-1]),flush=True);curves(history)
        save(OUT/'last.pt',payload()|dict(epoch=ep,validation=v))
        first=0;loss_sum=0.;seen=0;active=0.;checkpoint(ep+1,0)
    selected=torch.load(OUT/'candidate.pt',map_location='cpu',weights_only=False)
    atomic(OUT/'summary.json',dict(status='complete',epochs=EPOCHS,selected_epoch=selected['epoch'],
           validation=selected['validation'],normal_validation_only=True,independent_review=False,
           source_models_preserved=True,no_automatic_deployment=True,completed=stamp()))
    state('complete',epochs=EPOCHS,background_running=False,remaining_est_seconds=0,
          result='training/summary.json',independent_review=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['startup','train']);a=p.parse_args()
    startup() if a.action=='startup' else train()

