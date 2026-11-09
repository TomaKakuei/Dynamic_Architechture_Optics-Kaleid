"""Deployment interface: no simulated true state, OPD or WRMS argument."""
from common import *
from network import restore
import torch
class Controller:
 def __init__(self,checkpoint=None,device='cuda'):
  self.device=device;self.model,self.metadata=restore(checkpoint or HERE/'training/candidate.pt',device)
 @torch.no_grad()
 def predict(self,noisy_images):
  frames=np.asarray(noisy_images)
  if frames.ndim==3:frames=frames[None]
  if frames.shape[1:]!=(9,128,128) or not np.isfinite(frames).all():raise ValueError('Expected9field noisy sensor ADU')
  normalized=self.model.command(torch.tensor(frames,dtype=torch.float32,device=self.device)).cpu().numpy()
  physical=normalized.reshape(-1,8,5)*SCALES
  return dict(normalized=normalized,module2=physical[:,:3],module3=physical[:,3:6],module4=physical[:,6:],units=['mm','mm','mm','degree','degree'])
