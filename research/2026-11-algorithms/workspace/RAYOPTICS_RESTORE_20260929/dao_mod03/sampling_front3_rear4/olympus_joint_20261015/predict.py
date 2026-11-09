"""Image-only joint Olympus predictor. One full-system nine-field packet."""
from network import restore
import torch, numpy as np

class Predictor:
    def __init__(self,checkpoint=None,device='cuda'):
        from common import HERE
        self.model,self.metadata=restore(checkpoint or HERE/'training/candidate.pt',device)
        self.device=device
    @torch.inference_mode()
    def __call__(self,images):
        x=np.asarray(images)
        single=(x.ndim==3)
        if single:x=x[None]
        command=self.model.command(torch.as_tensor(x,dtype=torch.float32,device=self.device)).cpu().numpy()
        return command[0] if single else command

