"""Cached noisy-image training; center ABS plus signed edge DIFF physics."""
from local import *


class FullFieldObjective(Objective):
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.nominal,self.nominal_mask,_=self.full.phase(torch.zeros((1,35),device='cuda',dtype=torch.float64))

    def loss(self,raw,initial):
        command=-raw.clamp(-1,1);endpoint=initial.double()+command.double()
        mechanical=self.path_valid(endpoint,initial)
        selected=torch.where(mechanical)[0]
        physical=raw.sum()*0;invalid=int((~mechanical).sum());center_mean=0.;edge_mean=0.
        if len(selected):
            phase,pupil,chief=self.full.phase(endpoint[selected])
            mask=pupil&self.nominal_mask;weight=mask.double();design=self.full.low_design
            wave=torch.cat((phase[:,:1],phase[:,1:]-self.nominal[:,1:]),dim=1)
            gram=torch.einsum('nr,bfn,ns->bfrs',design,weight,design)
            rhs=torch.einsum('nr,bfn,bfn->bfr',design,wave,weight)
            coef=torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
            residual=torch.where(mask,wave-torch.einsum('nr,bfr->bfn',design,coef),0)
            variance=residual.square().sum(-1)/weight.sum(-1).clamp_min(1)
            good=chief.all(1)&(weight.mean(-1)>=.2).all(1)&torch.isfinite(variance).all(1)
            invalid+=int((~good).sum())
            if good.any():
                ell=torch.log1p(variance[good]/.07**2);bounded=ell/(1+ell)
                physical=(.5*bounded[:,0]+.5*bounded[:,1:].mean(1)).sum()
                center_mean=float(variance[good,0].clamp_min(0).sqrt().mean().detach())
                edge_mean=float(variance[good,1:].clamp_min(0).sqrt().mean().detach())
        physical=(physical+2*invalid)/len(raw)
        # Supervise raw outputs as well: a saturated output must retain a
        # gradient back toward the known training pose.
        pose=torch.nn.functional.smooth_l1_loss(raw,initial.float(),beta=.25)
        domain=torch.relu(endpoint.abs()-.95).square().sum(1).mean()
        total=physical+.2*pose+10*domain
        return total,dict(physical=float(physical.detach()),pose=float(pose.detach()),domain=float(domain.detach()),
                          invalid=invalid,center_mean=center_mean,edge_diff_mean=edge_mean)
