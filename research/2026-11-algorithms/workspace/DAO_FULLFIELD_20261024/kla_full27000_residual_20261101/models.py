"""Frozen full-data prior plus image-conditioned bounded residual correction."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0,str(SOURCE))
ARMS={'model_residual':'Frozen27000data epoch3 prior plus residual head trained on27000cases'}
_CACHE=None

class ResidualCorrector(nn.Module):
    cap=.05
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(9,32,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(64,96,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(4))
        self.head=nn.Sequential(nn.Linear(96*4*4+40,256),nn.GELU(),
                               nn.Linear(256,128),nn.GELU(),nn.Linear(128,40))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def command(self,images,context=None):
        if context is None:raise ValueError('Use Controller or provide observed-only model context')
        valid=context['available']
        residual=torch.where(valid[:,None,None,None],(images-context['expected'])/1024.,torch.zeros_like(images))
        features=self.encoder(residual).flatten(1)
        raw=self.head(torch.cat((features,context['prior']),1))
        delta=self.cap*torch.tanh(raw/self.cap)
        return (context['prior']+valid[:,None]*delta).clamp(-1.,1.)

def restore(path,arm='model_residual',device='cuda'):
    assert arm=='model_residual'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('experiment_arm') not in (None,'model_residual'):
        raise ValueError('Incompatible correction checkpoint')
    model=ResidualCorrector()
    if payload.get('experiment_arm')=='model_residual':model.load_state_dict(payload['model'])
    return model.to(device).eval(),payload

def cache_context(ids,device='cuda'):
    global _CACHE
    if _CACHE is None:
        folder=ROOT/'model_residual/camera_cache'
        _CACHE={key:np.load(folder/(key+'.npy'),mmap_mode='r') for key in
                ('lookup','expected','prior','available','done')}
    ids=np.asarray(ids)
    if (ids<0).any() or (ids>=len(_CACHE['lookup'])).any():raise ValueError('Invalid cache index')
    rows=_CACHE['lookup'][ids]
    if (rows<0).any() or not _CACHE['done'][rows].all():raise RuntimeError('Required camera context is not committed')
    return {k:torch.as_tensor(np.array(_CACHE[k][rows],copy=True),device=device)
            for k in ('expected','prior','available')}

class Controller:
    """Single noisy observation input, no true state or evaluation information."""
    def __init__(self,checkpoint,device='cuda'):
        from network import restore as restore_base
        from camera_context import ForwardCamera
        self.model,self.payload=restore(checkpoint,device=device)
        self.base,_=restore_base(self.payload['protocol']['initial_checkpoint'],device=device)
        self.camera=ForwardCamera();self.device=device

    @torch.no_grad()
    def predict(self,noisy_images):
        images=torch.as_tensor(noisy_images,device=self.device,dtype=torch.float32)
        if images.ndim==3:images=images[None]
        prior=self.base.command(images)
        expected,available=[],[]
        for command in prior.cpu().numpy():
            img,ok,_=self.camera.expected_or_fallback(-command.astype(np.float64))
            expected.append(img);available.append(ok)
        context=dict(prior=prior,expected=torch.as_tensor(np.stack(expected),device=self.device),
                     available=torch.as_tensor(available,device=self.device))
        return self.model.command(images,context=context)
