"""CPU-only directional derivative check; no real model/GPU or training."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import json
import torch
from trajectory_backward import four_steps
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
HERE=Path(__file__).resolve().parent


class ToyOptics:
    def __init__(self):
        generator=torch.Generator().manual_seed(20261025)
        self.a=torch.randn(7,4,generator=generator)*.3
        h=torch.randn(7,4,4,generator=generator)*.12
        self.h=.5*(h+h.transpose(1,2))
    def difference(self,x):
        return x@self.a.T+.5*torch.einsum('bi,fij,bj->bf',x,self.h,x)+.04*torch.sin(x@self.a.T)
    def response(self,x):
        return self.a[None]+torch.einsum('fij,bj->bfi',self.h,x)+.04*torch.cos(x@self.a.T)[:,:,None]*self.a[None]


def main():
    generator=torch.Generator().manual_seed(2026102521)
    optical=ToyOptics();truth=torch.randn(8,4,generator=generator)*.45
    measured=optical.difference(truth)+torch.randn(8,7,generator=generator)*.01
    target=torch.randn(8,4,generator=generator)*.2
    rows=[]
    for scale in (.02,.5,1.0):
        prior=(truth+scale*torch.randn(8,4,generator=generator)).clamp(-1.45,1.45).requires_grad_(True)
        result=four_steps(prior,measured,optical)
        loss=(result-target).square().mean()
        grad=torch.autograd.grad(loss,prior)[0]
        for kind in ('gradient','random'):
            direction=grad if kind=='gradient' else torch.randn(prior.shape,generator=generator)
            direction=direction/direction.square().mean().sqrt().clamp_min(1e-20)
            predicted=float((grad*direction).sum())
            for h in (1e-4,1e-5,1e-6):
                with torch.no_grad():
                    minus=(four_steps(prior-h*direction,measured,optical)-target).square().mean()
                    plus=(four_steps(prior+h*direction,measured,optical)-target).square().mean()
                observed=float((plus-minus)/(2*h))
                error=abs(observed-predicted)/max(1e-10,abs(observed),abs(predicted))
                rows.append(dict(prior_scale=scale,direction=kind,h=h,predicted=predicted,observed=observed,relative_error=error))
    finest=[r for r in rows if r['h']==1e-6]
    passed=all(r['relative_error']<2e-4 for r in finest)
    report=dict(status='passed' if passed else 'failed',device='cpu',optical_model='Synthetic smooth quadratic-plus-sine; not Nikon',
        optimizer_updates=0,max_finest_relative_error=max(r['relative_error'] for r in finest),directions=rows,
        real_nikon_forward_and_gradient_validation_pending=True)
    (HERE/'trajectory_cpu_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],max_finest_relative_error=report['max_finest_relative_error'])))
    assert passed


if __name__=='__main__':main()
