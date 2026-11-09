"""Small real-Nikon check of forward equivalence and four-step derivatives."""
from pathlib import Path
import sys,os,json,ctypes,traceback
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'original_routes_40ep_20261025'))
import worker as old
from worker import torch,np,exp,atomic
from trajectory_backward import four_steps
from train_four import FixedStatsNetwork,ExactTrajectoryNetwork


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        system=old.make_system('nikon');data=old.load_data(system)
        probe=json.loads((HERE/'diagnosis.json').read_text())['training_indices']
        ix=np.array(probe[:8]+probe[-8:])
        torch.manual_seed(2026102501);model,_=old.make_model(system)
        original=torch.load(old.HERE/'nikon/candidate.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(original['model']);del original
        model.__class__=FixedStatsNetwork;model.eval()
        image=torch.tensor(data['image'][ix],device='cuda')
        raw=torch.tensor(data['raw'][ix],device='cuda')
        target=torch.tensor(data['q'][ix],device='cuda',dtype=torch.float64)
        with torch.no_grad():
            neural=exp.LargerNetwork.forward(model,image,raw)
            prior=((neural*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5)).double()
            observed_prediction=model(image,raw)
            measured=raw.flatten(1).double()/7.
            physical=four_steps(prior,measured,model)
            new_prediction=((physical-model.base.target_mean)/model.base.target_std).float()
        difference=float((new_prediction-observed_prediction).abs().max())
        # Continuous physical loss removes FP32 output rounding from the derivative test.
        def objective(physical):
            fields,valid,_=system.metrics(target-physical,target)
            assert bool(valid.all())
            ell=torch.log1p((fields/.07).square());term=ell/(1+ell)
            control=(.5*term.mean(1)+.5*term.amax(1)).mean()
            estimate=(physical-model.base.target_mean.double())/model.base.target_std.double()
            label=(target-model.base.target_mean.double())/model.base.target_std.double()
            return control+.02*torch.nn.functional.smooth_l1_loss(estimate,label,beta=.25)
        variable=prior.clone().requires_grad_(True)
        loss=objective(four_steps(variable,measured,model))
        gradient=torch.autograd.grad(loss,variable)[0]
        # The old terminal-GN approximation provides a useful magnitude/direction comparison.
        with torch.no_grad():
            terminal=four_steps(prior,measured,model)
            jac=model.response(terminal)
            gram=jac.transpose(1,2)@jac+1e-4*torch.eye(35,device='cuda',dtype=torch.float64)
        old_prior=prior.clone().requires_grad_(True)
        implicit=torch.linalg.solve(gram,(1e-4*(old_prior-old_prior.detach())).unsqueeze(-1)).squeeze(-1)
        old_gradient=torch.autograd.grad(objective(terminal+implicit),old_prior)[0]
        generator=torch.Generator(device='cuda').manual_seed(2026102522)
        rows=[]
        for kind in ('exact_gradient','random'):
            direction=gradient if kind=='exact_gradient' else torch.randn(prior.shape,dtype=prior.dtype,device='cuda',generator=generator)
            direction=direction/direction.square().mean().sqrt().clamp_min(1e-20)
            predicted=float((gradient*direction).sum())
            for h in (1e-4,3e-5):
                with torch.no_grad():
                    minus=objective(four_steps(prior-h*direction,measured,model))
                    plus=objective(four_steps(prior+h*direction,measured,model))
                observed=float((plus-minus)/(2*h))
                rows.append(dict(direction=kind,h=h,predicted=predicted,observed=observed,
                    relative_error=abs(predicted-observed)/max(1e-8,abs(predicted),abs(observed)),
                    negative_step_change=float(minus-loss.detach())))
        passed=difference<1e-5 and all(r['relative_error']<.002 for r in rows if r['h']==3e-5)
        report=dict(status='passed' if passed else 'failed',training_indices=ix.tolist(),training_count=len(ix),
            validation_cases_used=0,optimizer_updates=0,forward_max_abs_difference=difference,
            loss=float(loss.detach()),directions=rows,
            implicit_vs_trajectory_gradient_cosine=float(torch.nn.functional.cosine_similarity(gradient.flatten(),old_gradient.flatten(),dim=0)),
            implicit_gradient_relative_difference=float((old_gradient-gradient).norm()/gradient.norm()),
            neural_parameters=sum(p.numel() for p in model.parameters()),
            scope='Chosen four-step smooth branch of actual Nikon forward; discrete branch boundaries not claimed differentiable')
        atomic(HERE/'trajectory_nikon_check.json',report);print(json.dumps(report),flush=True)
        assert passed,'Real Nikon trajectory check failed; do not train candidate'
    except Exception:
        atomic(HERE/'trajectory_nikon_failure.json',dict(error=traceback.format_exc(),updated=exp.stamp()));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':main()
