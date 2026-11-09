"""Training-only directional check of the approximate four-step backward."""
from stable import *
import os,time,traceback,ctypes
OUT=HERE/'gradient_check'
OUT.mkdir(exist_ok=True)

def state(status,**extra):
    exp.atomic(OUT/'state.json',dict(status=status,pid=os.getpid(),updated=exp.stamp(),**extra))

def main():
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        state('waiting_for_four_epoch_postprocessing')
        while True:
            previous=json.loads((HERE/'state.json').read_text())
            if previous['status']=='completed':break
            if previous['status']=='failed':raise RuntimeError('Complete or repair the four-epoch run before this diagnostic')
            time.sleep(10)
        state('loading_training_data')
        system=StableSystem();data=system.load_data()
        train=np.flatnonzero(data['train'])
        indices=np.random.default_rng(SEED+1).permutation(train)[:64]
        image=torch.as_tensor(data['image'][indices],device='cuda')
        raw=torch.as_tensor(data['raw'][indices],device='cuda')
        target=torch.as_tensor(data['q'][indices],dtype=torch.float32,device='cuda')
        report=dict(updated=exp.stamp(),training_indices=indices.tolist(),training_count=64,
            validation_labels_used=0,optimizer_updates=0,checkpoints={},
            scope='Directional finite changes of the actual same four-step forward versus its terminal implicit-GN backward; no training or selection on validation')
        model=system.model('large').eval()
        for checkpoint in ('initial','last'):
            if checkpoint=='last':model.load_state_dict(torch.load(HERE/'stable_training/last.pt',map_location='cpu',weights_only=False)['model'])
            with torch.no_grad():prior=NormalizedTrunk.forward(model,image,raw).detach().clone()
            override=[prior.clone().requires_grad_(True)]
            hook=model.head[-1].register_forward_hook(lambda module,args,output:override[0])
            try:
                loss,parts=system.loss(model,data,indices)
                direct=torch.nn.functional.smooth_l1_loss(override[0],(target-model.base.target_mean)/model.base.target_std,beta=.25)
                total_grad=torch.autograd.grad(loss,override[0],retain_graph=True)[0].detach()
                direct_grad=torch.autograd.grad(direct,override[0])[0].detach()
                optical_grad=total_grad-direct_grad
                baseline=float(loss.detach())
                override[0]=prior
                model.prior_prediction=prior
                with torch.no_grad():zero_loss,_=system.loss(model,data,indices)
                assert abs(float(zero_loss)-baseline)<1e-7
                result=dict(loss=baseline,zero_step_loss=float(zero_loss),parts=parts,gradients={},directions={})
                for name,gradient in [('total',total_grad),('direct_pose',direct_grad),('optical_plus_terminal_pose',optical_grad)]:
                    result['gradients'][name]=dict(norm=float(gradient.norm()),rms=float(gradient.square().mean().sqrt()))
                result['gradient_cosine_direct_optical']=float(torch.nn.functional.cosine_similarity(direct_grad.flatten(),optical_grad.flatten(),dim=0))
                for name,g in [('total',total_grad),('direct_pose',direct_grad),('optical_plus_terminal_pose',optical_grad)]:
                    direction=g/g.square().mean().sqrt().clamp_min(1e-12)
                    predicted=float((total_grad*direction).sum())
                    rows=[]
                    for step in (.01,.001,.0001):
                        state('directional_probe',checkpoint=checkpoint,direction=name,step=step)
                        values=[]
                        for sign in (-1,1):
                            override[0]=prior+sign*step*direction
                            # The training helper caches prior_prediction only
                            # while gradients are enabled. Refresh that cache
                            # for each numerical probe so the direct pose term
                            # is evaluated at the perturbed prior as well.
                            model.prior_prediction=override[0]
                            with torch.no_grad():value,terms=system.loss(model,data,indices)
                            values.append(float(value))
                        derivative=(values[1]-values[0])/(2*step)
                        rows.append(dict(step=step,loss_minus=values[0],loss_plus=values[1],
                            actual_central_derivative=derivative,predicted_derivative=predicted,
                            derivative_ratio=derivative/predicted if abs(predicted)>1e-12 else None,
                            negative_gradient_change=values[0]-baseline))
                    result['directions'][name]=rows
                    report['checkpoints'][checkpoint]=result
                    exp.atomic(OUT/'report.json',report)
                print(json.dumps(dict(checkpoint=checkpoint,loss=baseline,gradients=result['gradients'],directions=result['directions'])),flush=True)
            finally:hook.remove()
        state('completed',optimizer_updates=0,assistant_review='pending')
    except Exception:
        state('failed',error=traceback.format_exc());raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
