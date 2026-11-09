from optics import *
from geometry import Geometry
class Objective:
 def __init__(self,device='cuda'):self.center=Optics(grid=65,fields=[0],device=device);self.geometry=Geometry()
 def loss(self,command,initial):
  q=initial.double()+command.double();w,s=self.center.metrics(q);w=w[:,0];s=s[:,0]
  r=w/.07;ell=torch.log1p(r*r);physical=ell/(1+ell) # strictly below1: invalid states cannot be a cheaper destination
  paths=torch.tensor([self.geometry.path(b,a) for a,b in zip(initial.detach().cpu().numpy(),q.detach().cpu().numpy())],device=q.device)
  valid=(s>=.98)&torch.isfinite(w)&paths
  physical=torch.where(valid,physical,2+torch.nan_to_num(physical,nan=1.,posinf=1.))
  pose=torch.nn.functional.smooth_l1_loss(command,-initial.float(),reduction='none').mean(1)
  mechanical=self.center.mechanical_loss(q)
  loss=physical+.05*pose+10*mechanical
  return loss.mean(),dict(physical=float(physical.mean().detach()),pose=float(pose.mean().detach()),mechanical=float(mechanical.mean().detach()),wrms=float(w.mean().detach()))
