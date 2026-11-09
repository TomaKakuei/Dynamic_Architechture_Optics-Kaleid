"""Reproduce the first bad AdamW update; test conditioning fixes on training only."""
from pathlib import Path
import sys,os,time,traceback,json,ctypes
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'nikon_unified_repair_20261024'
sys.path.insert(0,str(PREVIOUS))
from repair import RepairSystem,FullPoseTrunk,exp,torch,np,SEED,PARAMETERS

def state(status,**extra):
    exp.atomic(HERE/'diagnostic_state.json',dict(status=status,pid=os.getpid(),updated=exp.stamp(),**extra))

@torch.no_grad()
def probe(system,model,data,indices):
    model.eval()
    image=torch.as_tensor(data['image'][indices],device='cuda')
    raw=torch.as_tensor(data['raw'][indices],device='cuda')
    pred=FullPoseTrunk.forward(model,image,raw)
    truth=torch.as_tensor(data['q'][indices],dtype=torch.float32,device='cuda')
    normalized=(truth-model.base.target_mean)/model.base.target_std
    physical=pred*model.base.target_std+model.base.target_mean
    endpoint,initial,command,_=system.forward(model,data,indices)
    f,v,a=system.metrics(endpoint,initial)
    return dict(neural_pose_mse=float((pred-normalized).square().mean()),
        prior_clipped_fraction=float((physical.abs()>1.5).float().mean()),
        prior_max_coordinate=float(physical.abs().max()),
        full_field=exp.aggregate(f.cpu().numpy(),v.cpu().numpy(),a.cpu().numpy(),.07)),pred.detach().cpu()

def first_step(system,model,data,train_batch,check_batch,lr,label,head_lr=None):
    before,pred_before=probe(system,model,data,check_batch)
    train_before,_=probe(system,model,data,train_batch)
    model.train();model.zero_grad(set_to_none=True)
    loss,parts=system.loss(model,data,train_batch)
    loss.backward()
    norms={}
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        group=name.split('.')[0]
        norms[group]=norms.get(group,0.)+float(p.grad.double().square().sum())
    norms={k:float(np.sqrt(v)) for k,v in norms.items()}
    original_grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
    if head_lr is None:optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    else:
        readout=list(model.head[-1].parameters());ids={id(p) for p in readout}
        optimizer=torch.optim.AdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=lr),
            dict(params=readout,lr=head_lr)],weight_decay=1e-4)
    optimizer.step()
    after,pred_after=probe(system,model,data,check_batch)
    train_after,_=probe(system,model,data,train_batch)
    model.train()
    loss_after,parts_after=system.loss(model,data,train_batch)
    result=dict(label=label,learning_rate=lr,readout_learning_rate=head_lr,discarded_optimizer_updates=1,
        training_loss_before=float(loss.detach()),training_loss_after=float(loss_after.detach()),
        loss_parts_before=parts,loss_parts_after=parts_after,gradient_norm_before_clip=original_grad,
        gradient_norm_by_module=norms,check_neural_prediction_delta_rms=float((pred_after-pred_before).square().mean().sqrt()),
        training_batch_before=train_before,training_batch_after=train_after,
        check_batch_before=before,check_batch_after=after)
    del loss,loss_after,optimizer
    model.zero_grad(set_to_none=True)
    return result

def main():
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        state('waiting_for_previous_run')
        while True:
            previous=json.loads((PREVIOUS/'state.json').read_text())
            if previous['status']=='completed':break
            if previous['status']=='failed':raise RuntimeError('Previous run failed; inspect it before concurrent GPU work')
            time.sleep(10)
        state('loading_training_data')
        system=RepairSystem();data=system.load_data();train=np.flatnonzero(data['train'])
        order=np.random.default_rng(SEED+1).permutation(train)
        train_batch=order[:128];check_batch=order[128:256]
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
        model=system.model('large').eval()
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        x=[]
        with torch.no_grad():
            for start in range(0,len(train),256):
                ix=train[start:start+256]
                image=torch.as_tensor(data['image'][ix],device='cuda')
                raw=torch.as_tensor(data['raw'][ix],device='cuda')
                x.append(model.features(image,raw).cpu())
        x=torch.cat(x).double()
        y=(torch.as_tensor(data['q'][train],dtype=torch.float64)-model.base.target_mean.cpu())/model.base.target_std.cpu()
        xm=x.mean(0);ym=y.mean(0);xs=x.std(0);xc=x-xm;yc=y-ym
        cov=xc.T@xc/len(x);rhs=xc.T@yc/len(x)
        stats=dict(feature_std_quantiles=np.quantile(xs.numpy(),[0,.01,.1,.25,.5,.75,.9,.99,1]).tolist(),
            features_below_1e6=int((xs<1e-6).sum()),features_below_1e3=int((xs<1e-3).sum()),
            features_below_005=int((xs<.05).sum()),
            original_weight_norm=float(initial['head.3.weight'].norm()),
            original_bias_norm=float(initial['head.3.bias'].norm()))
        candidates={}
        for name,regularizer in (
            ('raw_feature_ridge',.001*torch.ones(512,dtype=torch.float64)),
            ('variance_floor',.001*xs.clamp_min(.05).square())):
            beta=torch.linalg.solve(cov+torch.diag(regularizer),rhs)
            weight=beta.T.float();bias=(ym-xm@beta).float()
            prediction=x.float()@weight.T+bias
            weights=dict(initial);weights['head.3.weight']=weight;weights['head.3.bias']=bias
            candidates[name]=weights
            fit=dict(weight_norm=float(weight.norm()),weight_max=float(weight.abs().max()),
                bias_norm=float(bias.norm()),normalized_pose_training_mse=float((prediction-y).square().mean()),
                fit_samples=24000,validation_labels_used=0)
            stats[name]=fit
            exp.save_pt(HERE/(name+'_initialized.pt'),dict(model=weights,fit=fit,epoch=0))
        exp.atomic(HERE/'conditioning.json',stats)
        report=dict(updated=exp.stamp(),scope='256 training-only diagnostic samples, no validation-based tuning',
            training_batch=train_batch.tolist(),check_batch=check_batch.tolist(),conditioning=stats,probes={})
        for name,weights in [('old_unbounded',initial),*candidates.items()]:
            for lr in ([1e-4] if name=='old_unbounded' else [1e-4,1e-5,1e-6]):
                label=name+'_'+str(lr);state('first_update_probe',label=label)
                model.load_state_dict(weights)
                report['probes'][label]=first_step(system,model,data,train_batch,check_batch,lr,label)
                exp.atomic(HERE/'diagnosis.json',report)
                print(json.dumps(dict(label=label,conditioning=stats.get(name),
                    result=report['probes'][label])),flush=True)
        state('completed',discarded_optimizer_updates=7,full_training_epochs=0)
    except Exception:
        state('failed',error=traceback.format_exc());raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
