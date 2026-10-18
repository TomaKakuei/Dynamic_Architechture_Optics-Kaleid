"""One whole-system packet, two encoders, joint seven-body attention."""
import torch
from torch import nn
from .backbone import SpatialEncoder, GroupFactorizedHead


class ModularNetwork(SpatialEncoder):
    def __init__(self, groups):
        super().__init__(9, [5] * groups)
        y, x = torch.meshgrid((torch.arange(128)+.5)/64-1, (torch.arange(128)+.5)/64-1, indexing='ij')
        basis = torch.stack([torch.ones_like(x), x, y, x*y, x*x, y*y, torch.sin(torch.pi*x), torch.sin(torch.pi*y)])
        self.register_buffer('sensor_moments', basis)
        self.head = GroupFactorizedHead(256 + 72, [5] * groups)


class BranchEncoder(torch.nn.Module):
    def __init__(self, branch, payload):
        super().__init__()
        if branch not in ('front', 'rear'):
            raise ValueError(branch)
        self.branch = branch
        self.groups = 3 if branch == 'front' else 4
        self.base = ModularNetwork(self.groups)
        for key in ('target_mean', 'target_std', 'reference', 'noise_std'):
            self.register_buffer(key, torch.as_tensor(payload[key], dtype=torch.float32).clone())
        width = 128
        self.node_encoder = torch.nn.Sequential(torch.nn.Linear(68,64),torch.nn.GELU(),torch.nn.Linear(64,width),torch.nn.GELU())
        self.raw_global = torch.nn.Sequential(torch.nn.Linear(9*16*16,width),torch.nn.GELU(),torch.nn.Linear(width,width))
        self.feature_projection = torch.nn.Linear(256+72,width)
        self.pose_projections = torch.nn.ModuleList([torch.nn.Linear(5,width) for _ in range(self.groups)])
        self.module_identity = torch.nn.Parameter(torch.zeros(1,self.groups,width))
        self.cross_norm = torch.nn.LayerNorm(width)
        self.node_attention = torch.nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.module_norm = torch.nn.LayerNorm(width)
        self.module_attention = torch.nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.ff_norm = torch.nn.LayerNorm(width)
        self.ff = torch.nn.Sequential(torch.nn.Linear(width,256),torch.nn.GELU(),torch.nn.Linear(256,width))
        self.correction_heads = torch.nn.ModuleList([torch.nn.Linear(width,5) for _ in range(self.groups)])
        for head in self.correction_heads:
            torch.nn.init.zeros_(head.weight)
            torch.nn.init.zeros_(head.bias)
        y,x = torch.meshgrid((torch.arange(16)+.5)/8-1,(torch.arange(16)+.5)/8-1,indexing='ij')
        xy = torch.stack((x.flatten(),y.flatten()),1).repeat(9,1)
        field = torch.repeat_interleave(torch.linspace(-1,1,9),256)[:,None]
        self.register_buffer('coordinates',torch.cat((xy,field,xy.square().sum(1,keepdim=True)),1))

    def features(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (9,128,128):
            raise ValueError('Expected batch x9x128x128 noisy sensor ADU')
        x = torch.asinh((images-self.reference)/self.noise_std/3.)
        feat = self.base.stage3(self.base.stage2(self.base.stage1(self.base.stem(x))))
        moments = torch.einsum('bfxy,kxy->bfk',x,self.base.sensor_moments)/16384.
        feat = torch.cat((self.base.pool(feat).flatten(1),moments.flatten(1)),1)
        estimate = self.base.head(feat)
        raw = (images-self.reference)/16384.
        patches = raw.unfold(2,8,8).unfold(3,8,8).reshape(len(raw),9*16*16,64)
        nodes = self.node_encoder(torch.cat((patches,self.coordinates[None].expand(len(raw),-1,-1)),2))
        context = torch.stack([proj(estimate[:,5*i:5*i+5]) for i,proj in enumerate(self.pose_projections)],1)
        context = context+self.feature_projection(feat)[:,None]+self.raw_global(patches.mean(2))[:,None]+self.module_identity
        context = context+self.node_attention(self.cross_norm(context),nodes,nodes,need_weights=False)[0]
        normalized = self.module_norm(context)
        context = context+self.module_attention(normalized,normalized,normalized,need_weights=False)[0]
        context = context+self.ff(self.ff_norm(context))
        correction = torch.cat([head(context[:,i]) for i,head in enumerate(self.correction_heads)],1)
        return estimate+correction, context

    def forward(self, images):
        return self.features(images)[0]

    def physical_prediction(self, images):
        return self(images)*self.target_std+self.target_mean

    def command(self, images):
        return -self.physical_prediction(images).clamp(-1.,1.)

    def train(self, mode=True):
        super().train(mode)
        for layer in self.base.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self


class OlympusModel(nn.Module):
    def __init__(self, calibrations):
        super().__init__()
        self.front = BranchEncoder('front', calibrations['front'])
        self.rear = BranchEncoder('rear', calibrations['rear'])
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
