"""Four-step GN trajectory VJP using native optical Hessian-vector products.

This is a research candidate, not yet validated on the real Nikon model.
The forward solve/line search is the existing exact-response solve. Reverse
mode keeps its chosen gain/clamp branch fixed, as ordinary autodiff does for
these discrete branches. It includes dJ/dx via two scalar optical HVPs and
avoids materializing an observation-by-coordinate-by-coordinate Hessian.
"""
import torch


class TrajectoryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, physical, prior, measured, optical):
        ridge=1e-4
        residual=optical.difference(physical)-measured
        jac=optical.response(physical);jt=jac.transpose(1,2)
        eye=torch.eye(physical.shape[1],dtype=physical.dtype,device=physical.device)
        gram=jt@jac+ridge*eye
        rhs=(jt@residual.unsqueeze(-1)).squeeze(-1)+ridge*(physical-prior)
        delta=-torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
        maximum,index=delta.abs().max(1,keepdim=True)
        divisor=torch.clamp(maximum/.25,min=1.)
        step=delta/divisor
        objective=residual.square().sum(1)+ridge*(physical-prior).square().sum(1)
        accepted=torch.zeros(len(physical),dtype=torch.bool,device=physical.device)
        gain_chosen=torch.zeros_like(maximum)
        clamp_mask=torch.ones_like(physical)
        result=physical.clone()
        for gain in (1.,.5,.25):
            rows=torch.nonzero(~accepted,as_tuple=True)[0]
            if not len(rows):break
            unconstrained=physical[rows]+gain*step[rows]
            trial=unconstrained.clamp(-1.5,1.5)
            innovation=optical.difference(trial)-measured[rows]
            merit=innovation.square().sum(1)+ridge*(trial-prior[rows]).square().sum(1)
            good=merit<objective[rows]
            if bool(good.any()):
                selected=rows[good]
                result=result.index_copy(0,selected,trial[good])
                accepted[selected]=True
                gain_chosen[selected]=gain
                clamp_mask[selected]=((unconstrained[good]>=-1.5)&(unconstrained[good]<=1.5)).to(physical.dtype)
        if getattr(optical,'trajectory_trace',None) is not None:
            optical.trajectory_trace.append(dict(gain=gain_chosen.detach().cpu().numpy(),
                cap_index=index.detach().cpu().numpy(),cap_active=(maximum>.25).detach().cpu().numpy(),
                clamp_mask=clamp_mask.detach().cpu().numpy(),physical=result.detach().cpu().numpy()))
        ctx.optical=optical
        ctx.save_for_backward(physical,jac,gram,delta,residual,maximum,index,divisor,gain_chosen,clamp_mask)
        return result

    @staticmethod
    def backward(ctx, upstream):
        physical,jac,gram,delta,residual,maximum,index,divisor,gain,clamp_mask=ctx.saved_tensors
        # VJP through endpoint clamp, accepted gain and infinity-norm step cap.
        direct=upstream*clamp_mask
        capped=direct*gain
        step_vjp=capped/divisor
        cap_active=maximum>.25
        correction=(capped*delta).sum(1,keepdim=True)/(divisor*maximum.clamp_min(1e-30))
        signed_maximum=delta.gather(1,index).sign()
        cap_term=torch.zeros_like(delta).scatter(1,index,correction*signed_maximum)
        step_vjp=step_vjp-torch.where(cap_active,cap_term,torch.zeros_like(cap_term))
        adjoint=torch.linalg.solve(gram,step_vjp.unsqueeze(-1)).squeeze(-1)
        jac_delta=(jac@delta.unsqueeze(-1)).squeeze(-1)
        jac_adjoint=(jac@adjoint.unsqueeze(-1)).squeeze(-1)
        # If delta=-A^-1 g, then
        # (D_x delta)' v = -v - H_{r+J delta} A^-T v - H_{J A^-T v} delta.
        # Each weighted optical Hessian is applied to one vector only.
        with torch.enable_grad():
            x=physical.detach().requires_grad_(True)
            observation=ctx.optical.difference(x)
            grad1=torch.autograd.grad((observation*(residual+jac_delta).detach()).sum(),x,
                                      create_graph=True,retain_graph=True)[0]
            hvp1=torch.autograd.grad((grad1*adjoint.detach()).sum(),x,retain_graph=True)[0]
            grad2=torch.autograd.grad((observation*jac_adjoint.detach()).sum(),x,
                                      create_graph=True,retain_graph=True)[0]
            hvp2=torch.autograd.grad((grad2*delta.detach()).sum(),x)[0]
        return direct-step_vjp-hvp1-hvp2,1e-4*adjoint,jac_adjoint,None


def four_steps(prior,measured,optical):
    physical=prior
    for _ in range(4):physical=TrajectoryStep.apply(physical,prior,measured,optical)
    return physical
