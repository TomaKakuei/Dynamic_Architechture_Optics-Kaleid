"""Bounded training-only investigation of the failed enlarged Nikon run."""
from pathlib import Path
import sys,os,json,gc,traceback,ctypes
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'original_routes_40ep_20261025'))
import worker as old
from worker import torch,np,exp,atomic
ORIGINAL=exp.LargerNetwork.forward


def intercepted(self,image,raw=None):
    return self.probe_prior if hasattr(self,'probe_prior') else ORIGINAL(self,image,raw)


def state(status,**kw):
    atomic(HERE/'diagnosis_state.json',dict(status=status,pid=os.getpid(),updated=exp.stamp(),**kw))


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    exp.LargerNetwork.forward=intercepted
    try:
        state('loading')
        system=old.make_system('nikon');data=old.load_data(system)
        rng=np.random.default_rng(2026102517)
        old_ids=np.flatnonzero(data['train']&(np.arange(len(data['q']))<30000))
        new_ids=np.arange(30000,34000)
        ix=np.r_[rng.choice(old_ids,96,replace=False),rng.choice(new_ids,32,replace=False)]
        torch.manual_seed(2026102501)
        model,_=old.make_model(system)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        image=torch.tensor(data['image'][ix],device='cuda');raw=torch.tensor(data['raw'][ix],device='cuda')
        report=dict(training_indices=ix.tolist(),validation_cases_used=0,probe_optimizer_updates_discarded=0,
            parameters=sum(p.numel() for p in model.parameters()),checkpoints={},update_probes=[])
        report['mode_sensitive_layers']=[dict(name=n,type=type(m).__name__) for n,m in model.named_modules()
            if isinstance(m,(torch.nn.modules.batchnorm._BatchNorm,torch.nn.Dropout))]
        for checkpoint in ('initial','paused_epoch8'):
            state('checkpoint_diagnosis',checkpoint=checkpoint)
            if checkpoint!='initial':
                payload=torch.load(old.HERE/'nikon/resume.pt',map_location='cpu',weights_only=False,mmap=True)
                model.load_state_dict(payload['model']);del payload
            model.eval()
            with torch.no_grad():
                prior=ORIGINAL(model,image,raw).detach().clone()
                physical=prior*model.base.target_std+model.base.target_mean
                terminal,_,command,_=system.forward(model,data,ix)
                fields,valid,_=system.metrics(terminal,torch.tensor(data['q'][ix],device='cuda'))
                eval_loss,parts=system.loss(model,data,ix)
                direct=torch.nn.functional.smooth_l1_loss(prior,
                    (torch.tensor(data['q'][ix],device='cuda')-model.base.target_mean)/model.base.target_std,beta=.25)
            row=dict(eval_loss=float(eval_loss),parts=parts,prior_pose_loss=float(direct),
                prior_rms=float(prior.square().mean().sqrt()),clamped_prior_fraction=float((physical.abs()>=1.5).float().mean()),
                train_probe_passes=int(((fields.max(1).values<.07)&valid).sum()),field_mean=float(fields.mean()))
            # Separate actual train-mode numerics from evaluation mode.
            model.train()
            with torch.no_grad():
                train_prior=ORIGINAL(model,image,raw)
                train_loss,_=system.loss(model,data,ix)
            row.update(train_loss=float(train_loss),train_eval_prior_max_diff=float((train_prior-prior).abs().max()))
            model.eval()
            # Directly check the neural-output derivative of the actual 4-step forward.
            model.probe_prior=prior.clone().requires_grad_(True)
            loss,_=system.loss(model,data,ix)
            gradient=torch.autograd.grad(loss,model.probe_prior)[0].detach()
            direction=gradient/gradient.square().mean().sqrt().clamp_min(1e-16)
            predicted=float((gradient*direction).sum())
            row['prior_gradient_norm']=float(gradient.norm());row['directions']=[]
            for step in (1e-3,1e-4):
                values=[]
                for sign in (-1,1):
                    model.probe_prior=prior+sign*step*direction
                    with torch.no_grad():value,_=system.loss(model,data,ix)
                    values.append(float(value))
                row['directions'].append(dict(step=step,loss_minus=values[0],loss_plus=values[1],
                    predicted_derivative=predicted,actual_derivative=(values[1]-values[0])/(2*step),
                    negative_gradient_change=values[0]-float(loss)))
            del model.probe_prior
            report['checkpoints'][checkpoint]=row
            atomic(HERE/'diagnosis.json',report)
        # Two matched single-update probes: the old LR versus 10x lower LR.
        for scale in (1.,.1):
            state('single_update',learning_rate_scale=scale)
            model.load_state_dict(initial);model.train()
            optimizer=exp.make_optimizer(model)
            for group in optimizer.param_groups:group['lr']*=scale
            optimizer.zero_grad(set_to_none=True)
            loss,parts=system.loss(model,data,ix);loss.backward()
            gradient_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
            optimizer.step();model.eval()
            with torch.no_grad():
                after=ORIGINAL(model,image,raw)
                value,after_parts=system.loss(model,data,ix)
                model.load_state_dict(initial)
                before=ORIGINAL(model,image,raw)
            report['update_probes'].append(dict(learning_rate_scale=scale,loss_before=float(loss.detach()),
                loss_after=float(value),gradient_norm=gradient_norm,
                neural_output_change_rms=float((after-before).square().mean().sqrt()),
                neural_output_change_max=float((after-before).abs().max()),parts_after=after_parts))
            report['probe_optimizer_updates_discarded']+=1
            del optimizer;gc.collect()
            atomic(HERE/'diagnosis.json',report)
        state('completed',assistant_review='pending')
        print(json.dumps(report),flush=True)
    except Exception:
        state('failed',error=traceback.format_exc());raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
