from common import *
import numpy as np
import torch
from tensor_optics import TensorOptics, SamplingOptics

class Objective:
    def __init__(self, device='cuda'):
        self.engine=SamplingOptics()
        self.center=TensorOptics(self.engine,pupil_grid=64,device=device)
        for name in ('launch_points','launch_directions','reference_centers','reference_exits','reference_opls'):
            setattr(self.center,name,getattr(self.center,name)[:1].clone())
        self.full=TensorOptics(self.engine,pupil_grid=64,device=device)

    def path_valid(self, q, initial):
        result=[]
        for end,start in zip(q.detach().cpu().numpy(),initial.detach().cpu().numpy()):
            ok=np.isfinite(end).all() and abs(end).max()<=1.
            if ok:ok=bool(self.engine.geometry.path(end,start=start)['safe'])
            result.append(ok)
        return torch.tensor(result,dtype=torch.bool,device=q.device)

    def loss(self, command, initial):
        q=initial.double()+command.double()
        mechanical=self.path_valid(q,initial)
        selected=torch.where(mechanical)[0]
        physical=command.sum()*0.
        invalid=int((~mechanical).sum())
        mean_wrms=0.
        if len(selected):
            # One production trace; no independent implementation or replay.
            variance,_,pupil,chief=self.center.high_order(q[selected])
            good=chief.all(1)&(pupil.double().mean(-1)>=.2).all(1)&torch.isfinite(variance).all(1)
            invalid+=int((~good).sum())
            if good.any():
                w=variance[good,0].clamp_min(1e-20).sqrt()
                ell=torch.log1p((w/.07).square())
                physical=(ell/(1+ell)).sum()
                mean_wrms=float(w.mean().detach())
        physical=(physical+2.*invalid)/len(q)
        pose=torch.nn.functional.smooth_l1_loss(command,-initial.float())
        domain=torch.relu(q.abs()-1.).square().sum(1).mean()
        loss=physical+.05*pose+10.*domain
        return loss,dict(physical=float(physical.detach()),pose=float(pose.detach()),
                         domain=float(domain.detach()),invalid=invalid,center_valid_mean=mean_wrms)

    @torch.no_grad()
    def metrics(self, model, images, initial):
        model.eval()
        command=model.command(images).double()
        q=initial.double()+command
        mechanical=self.path_valid(q,initial)
        wrms=np.full((len(q),9),np.nan)
        valid=np.zeros(len(q),bool)
        selected=torch.where(mechanical)[0]
        if len(selected):
            variance,_,pupil,chief=self.full.high_order(q[selected])
            w=variance.clamp_min(0).sqrt()
            good=chief.all(1)&(pupil.double().mean(-1)>=.2).all(1)&torch.isfinite(w).all(1)
            ids=selected.cpu().numpy()
            wrms[ids]=w.cpu().numpy()
            valid[ids]=good.cpu().numpy()
        return wrms,valid,command.cpu().numpy()

