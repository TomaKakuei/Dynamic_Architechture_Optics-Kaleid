"""Three prospective improvements; all retain the original recovery loss."""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0, str(SOURCE))
from network import JointNetwork

ARMS = {
    'spatial_head': 'Preserve4x4 pretrained CNN spatial features through a zero-initialized residual readout',
    'model_residual': 'Frozen prior plus bounded learned correction conditioned on observed-minus-predicted camera images',
    'full_data': 'Original architecture, full27000training samples for3epochs',
}


class CoreCandidate(nn.Module):
    def __init__(self, core, arm):
        super().__init__()
        self.core, self.arm = core, arm
        self.spatial = None
        if arm == 'spatial_head':
            self.spatial = nn.Sequential(nn.Linear(256*4*4, 128), nn.GELU(), nn.Linear(128, 40))
            nn.init.zeros_(self.spatial[-1].weight); nn.init.zeros_(self.spatial[-1].bias)

    def command(self, images, context=None):
        if self.spatial is None:
            return self.core.command(images)
        # Same pretrained computation, with one extra readout BEFORE global pooling.
        c = self.core
        raw = (images-c.reference)/1024.
        fmap = c.base.stage3(c.base.stage2(c.base.stage1(c.base.stem(torch.asinh(raw)))))
        feat = c.base.pool(fmap).flatten(1)
        base = c.base.head(feat)
        patches = raw.unfold(2,8,8).unfold(3,8,8).contiguous().reshape(len(raw),2304,64)
        nodes = c.node_encoder(torch.cat((patches,c.coordinates[None].expand(len(raw),-1,-1)),2))
        latent = torch.stack([proj(base[:,5*i:5*i+5]) for i,proj in enumerate(c.pose_projections)],1)
        latent = latent+c.identity+c.project(feat)[:,None]+c.global_raw(patches.mean(-1))[:,None]
        latent = latent+c.node_attention(c.norm1(latent),nodes,nodes,need_weights=False)[0]
        n = c.norm2(latent)
        latent = latent+c.module_attention(n,n,n,need_weights=False)[0]
        latent = latent+c.ff(c.norm3(latent))
        logits = base+torch.cat([h(latent[:,i]) for i,h in enumerate(c.heads)],1)
        return -torch.tanh(logits+self.spatial(F.adaptive_avg_pool2d(fmap,4).flatten(1)))


class ResidualCorrector(nn.Module):
    """Training receives cached model-only context; deployment computes it from I."""
    cap = .05

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(9,32,5,stride=2,padding=2),nn.GELU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),nn.GELU(),
            nn.Conv2d(64,96,3,stride=2,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d(4))
        self.head = nn.Sequential(nn.Linear(96*4*4+40,256),nn.GELU(),
                                  nn.Linear(256,128),nn.GELU(),nn.Linear(128,40))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def command(self, images, context=None):
        if context is None:
            raise ValueError('Observed-only camera/prior context required; use Controller for deployment')
        available = context['available']
        residual = torch.where(available[:,None,None,None],
                               (images-context['expected'])/1024.,torch.zeros_like(images))
        features = self.encoder(residual).flatten(1)
        raw = self.head(torch.cat((features,context['prior']),1))
        delta = self.cap*torch.tanh(raw/self.cap)
        return (context['prior']+available[:,None]*delta).clamp(-1.,1.)


def restore(path, arm=None, device='cuda'):
    payload = torch.load(path,map_location='cpu',weights_only=False)
    saved = payload.get('experiment_arm')
    if saved is not None:
        if arm is not None and saved != arm:raise ValueError('Experiment/checkpoint mismatch')
        arm = saved
    assert arm in ARMS
    if arm == 'model_residual':
        model = ResidualCorrector()
    else:
        core = JointNetwork(payload['reference'])
        if saved is None:core.load_state_dict(payload['model'])
        model = CoreCandidate(core,arm)
    if saved is not None:model.load_state_dict(payload['model'])
    return model.to(device).eval(),payload


def cache_context(ids, device='cuda'):
    rows = []
    for case in ids:
        with np.load(ROOT/'model_residual/camera_cache'/f'{int(case)}.npz') as z:
            rows.append({k:z[k].copy() for k in ('expected','prior','available')})
    return {k:torch.as_tensor(np.stack([r[k] for r in rows]),device=device)
            for k in rows[0]}


class Controller:
    """Only noisy9field observations enter inference; no labels or case IDs."""
    def __init__(self, checkpoint, device='cuda'):
        self.model,self.payload = restore(checkpoint,device=device)
        self.device = device
        self.arm = self.payload['experiment_arm']
        if self.arm == 'model_residual':
            from network import restore as restore_base
            from camera_context import ForwardCamera
            self.base,_ = restore_base(self.payload['protocol']['initial_checkpoint'],device=device)
            self.camera = ForwardCamera()

    @torch.no_grad()
    def predict(self, noisy_images):
        x = torch.as_tensor(noisy_images,device=self.device,dtype=torch.float32)
        if x.ndim == 3:x = x[None]
        if self.arm != 'model_residual':return self.model.command(x)
        prior = self.base.command(x)
        packets,valid = [],[]
        for command in prior.cpu().numpy():
            packet,ok,_ = self.camera.expected_or_fallback(-command.astype(np.float64))
            packets.append(packet);valid.append(ok)
        context = dict(prior=prior,expected=torch.as_tensor(np.stack(packets),device=self.device),
                       available=torch.as_tensor(valid,device=self.device))
        return self.model.command(x,context=context)
