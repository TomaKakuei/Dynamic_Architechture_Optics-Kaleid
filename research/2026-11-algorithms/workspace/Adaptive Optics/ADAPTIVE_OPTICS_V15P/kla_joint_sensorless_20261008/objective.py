"""Nonlinear nine-field wavefront loss; Q2000 determines checkpoint quality.

Smooth wavefront residuals provide gradients even where highly aberrated
MTF is oscillatory and undersampled. No ideal quantities enter the predictor.
Defocus is retained, as required by the fixed-image-plane MTF objective.
"""
from optics import *
from geometry import Geometry
class Objective:
 def __init__(self,device='cuda'):
  self.optics=Optics(grid=33,device=device);self.geometry=Geometry()
 def loss(self,command,initial):
  q=initial.double()+command.double();phase,mask=self.optics.residual(q);w=mask.double()
  rms=torch.sqrt((phase.square()*w).sum(-1)/w.sum(-1).clamp_min(1)+1e-20)
  ell=torch.log1p((rms/.07).square());physical=(ell/(1+ell)).mean(1)
  paths=torch.tensor([self.geometry.path(b,a) for a,b in zip(initial.detach().cpu().numpy(),q.detach().cpu().numpy())],device=q.device)
  valid=(w.mean(-1)>=.98).all(-1)&torch.isfinite(rms).all(-1)&paths
  physical=torch.where(valid,physical,2+torch.nan_to_num(physical,nan=1.,posinf=1.))
  pose=torch.nn.functional.smooth_l1_loss(command,-initial.float(),reduction='none').mean(1)
  mechanical=self.geometry.penalty(q);loss=(physical+.05*pose+10*mechanical).mean()
  return loss,dict(physical=float(physical.detach().mean()),pose=float(pose.detach().mean()),mechanical=float(mechanical.detach().mean()),wrms=float(rms.detach().mean()))
