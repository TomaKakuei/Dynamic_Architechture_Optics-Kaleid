"""Separate BN running-statistic drift from learned parameter changes."""
from diagnose import *


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        system=old.make_system('nikon');data=old.load_data(system)
        ix=np.array(json.loads((HERE/'diagnosis.json').read_text())['training_indices'])
        torch.manual_seed(2026102501);model,_=old.make_model(system)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        payload=torch.load(old.HERE/'nikon/resume.pt',map_location='cpu',weights_only=False,mmap=True)
        final=payload['model']
        bnkeys=[k for k in initial if k.endswith(('running_mean','running_var','num_batches_tracked'))]
        image=torch.tensor(data['image'][ix],device='cuda');raw=torch.tensor(data['raw'][ix],device='cuda')
        result=dict(training_count=len(ix),validation_cases_used=0,normalization_layers=len(bnkeys)//3,
            all_neural_parameters_trainable=True,counterfactuals={})
        for weights,stats in [('initial','initial'),('initial','epoch8'),('epoch8','initial'),('epoch8','epoch8')]:
            state('normalization_counterfactual',weights=weights,statistics=stats)
            base=initial if weights=='initial' else final
            buffers=initial if stats=='initial' else final
            model.load_state_dict({k:(buffers[k] if k in bnkeys else v) for k,v in base.items()})
            model.eval()
            with torch.no_grad():
                loss,parts=system.loss(model,data,ix)
                endpoint,q,_,_=system.forward(model,data,ix)
                fields,valid,_=system.metrics(endpoint,q)
                prediction=ORIGINAL(model,image,raw)
            result['counterfactuals'][weights+'_weights_'+stats+'_stats']=dict(loss=float(loss),
                passes=int(((fields.max(1).values<.07)&valid).sum()),field_mean=float(fields.mean()),
                prior_rms=float(prediction.square().mean().sqrt()))
        # The proposed fix freezes only running statistics, not trainable BN affine parameters.
        model.load_state_dict(initial);model.eval()
        with torch.no_grad():before=ORIGINAL(model,image,raw)
        model.train()
        bnmodules=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
        for module in bnmodules:module.eval()
        with torch.no_grad():after=ORIGINAL(model,image,raw)
        result['fixed_train_eval_prior_max_diff']=float((after-before).abs().max())
        result['affine_parameters_remain_trainable']=all(m.weight.requires_grad and m.bias.requires_grad for m in bnmodules)
        # One real optimizer proposal on the same training batch, discarded after diagnosis.
        optimizer=exp.make_optimizer(model);optimizer.zero_grad(set_to_none=True)
        loss,_=system.loss(model,data,ix);loss.backward()
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True))
        optimizer.step()
        with torch.no_grad():value,parts=system.loss(model,data,ix)
        result['fixed_stats_original_lr_update']=dict(loss_before=float(loss.detach()),loss_after=float(value),gradient_norm=norm)
        result['running_buffers_unchanged']=all(torch.equal(model.state_dict()[k].cpu(),initial[k]) for k in bnkeys)
        atomic(HERE/'normalization_check.json',result)
        print(json.dumps(result),flush=True)
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
