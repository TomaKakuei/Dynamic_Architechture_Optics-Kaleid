"""Separate activation conditioning from the Adam step size, training data only."""
from diagnose import *

def main():
    assert json.loads((HERE/'diagnostic_state.json').read_text())['status']=='completed'
    system=RepairSystem();data=system.load_data();train=np.flatnonzero(data['train'])
    order=np.random.default_rng(SEED+1).permutation(train);batch=order[:128];check=order[128:256]
    model=system.model('large');initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    report=dict(scope='Training-only controls; all diagnostic optimizer updates discarded',probes={})
    for lr in (1e-6,1e-7,1e-8):
        model.load_state_dict(initial)
        name='original_head_'+str(lr)
        report['probes'][name]=first_step(system,model,data,batch,check,lr,name)
        exp.atomic(HERE/'head_diagnosis.json',report)
        print(json.dumps(dict(name=name,loss_before=report['probes'][name]['training_loss_before'],
            loss_after=report['probes'][name]['training_loss_after'])),flush=True)
    model.load_state_dict(initial)
    model.head[2]=torch.nn.LeakyReLU(.1)
    model.eval();features=[]
    with torch.no_grad():
        for start in range(0,len(train),256):
            ix=train[start:start+256]
            features.append(model.features(torch.as_tensor(data['image'][ix],device='cuda'),
                torch.as_tensor(data['raw'][ix],device='cuda')).cpu())
    x=torch.cat(features).double()
    y=(torch.as_tensor(data['q'][train],dtype=torch.float64)-model.base.target_mean.cpu())/model.base.target_std.cpu()
    xm=x.mean(0);ym=y.mean(0);xs=x.std(0)
    scale=xs.clamp_min(.01)
    z=(x-xm)/scale
    beta=torch.linalg.solve(z.T@z/len(z)+.001*torch.eye(512,dtype=torch.float64),z.T@(y-ym)/len(z))
    weight=(beta/scale[:,None]).T.float();bias=(ym-xm@(beta/scale[:,None])).float()
    fitted=dict(initial);fitted['head.3.weight']=weight;fitted['head.3.bias']=bias
    fit=dict(activation='LeakyReLU(0.1)',ridge=.001,feature_std_floor=.01,
        feature_std_quantiles=np.quantile(xs.numpy(),[0,.01,.1,.5,.9,.99,1]).tolist(),
        weight_norm=float(weight.norm()),weight_max=float(weight.abs().max()),bias_norm=float(bias.norm()),
        normalized_pose_training_mse=float(((x.float()@weight.T+bias)-y).square().mean()),
        fit_samples=24000,validation_labels_used=0)
    report['fit']=fit
    exp.save_pt(HERE/'leaky_initialized.pt',dict(model=fitted,fit=fit,epoch=0))
    for lr in (1e-4,1e-5,1e-6):
        model.load_state_dict(fitted)
        name='leaky_'+str(lr)
        report['probes'][name]=first_step(system,model,data,batch,check,lr,name)
        exp.atomic(HERE/'head_diagnosis.json',report)
        print(json.dumps(dict(name=name,loss_before=report['probes'][name]['training_loss_before'],
            loss_after=report['probes'][name]['training_loss_after'])),flush=True)
    exp.atomic(HERE/'head_diagnostic_completed.json',dict(updated=exp.stamp(),discarded_optimizer_updates=6))

if __name__=='__main__':main()
