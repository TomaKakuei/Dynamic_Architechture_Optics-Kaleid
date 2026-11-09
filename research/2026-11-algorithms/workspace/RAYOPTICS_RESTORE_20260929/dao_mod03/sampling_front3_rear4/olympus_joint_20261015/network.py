"""Two reused encoders share one whole-system observation and seven-body attention."""
from common import *
from legacy_branch import CoordinatedSpeckle
import torch
from torch import nn

class JointNetwork(nn.Module):
    def __init__(self, calibrations):
        super().__init__()
        self.front = CoordinatedSpeckle('front', calibrations['front'])
        self.rear = CoordinatedSpeckle('rear', calibrations['rear'])
        self.norm = nn.LayerNorm(128)
        self.attention = nn.MultiheadAttention(128,4,batch_first=True)
        self.ff_norm = nn.LayerNorm(128)
        self.ff = nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
        self.identity = nn.Parameter(torch.randn(1,7,128)*.01)
        self.heads = nn.ModuleList([nn.Linear(128,5) for _ in range(7)])
        for h in self.heads:
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)

    def forward(self, images):
        front, ft = self.front.features(images)
        rear, rt = self.rear.features(images)
        context = torch.cat((ft,rt),1) + self.identity
        n = self.norm(context)
        context = context+self.attention(n,n,n,need_weights=False)[0]
        context = context+self.ff(self.ff_norm(context))
        correction = torch.cat([h(context[:,i]) for i,h in enumerate(self.heads)],1)
        base = torch.cat((front*self.front.target_std+self.front.target_mean,
                          rear*self.rear.target_std+self.rear.target_mean),1)
        return base + correction

    def command(self, images):
        return -self(images).clamp(-1,1)

def initialize():
    payloads = {b:torch.load(PREVIOUS/'training'/b/'candidate.pt', map_location='cpu',weights_only=False) for b in BRANCHES}
    calibrations = {b:payloads[b]['calibration'] for b in BRANCHES}
    model = JointNetwork(calibrations)
    for b in BRANCHES:
        getattr(model,b).load_state_dict(payloads[b]['model'])
    return model, calibrations

def restore(path, device='cuda'):
    p = torch.load(path,map_location='cpu',weights_only=False)
    m = JointNetwork(p['calibrations'])
    m.load_state_dict(p['model'])
    return m.to(device).eval(), p

