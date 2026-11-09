"""CNN + raw noisy patch nodes + cross-module attention; one joint40D action."""
from common import *
import torch
from torch import nn
import importlib.util
_file=HERE.parent/'canonical_three_phase_universal/models.py'
_spec=importlib.util.spec_from_file_location('kla_architecture_source',_file);_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_module)
class JointNetwork(nn.Module):
 def __init__(self,reference):
  super().__init__();self.base=_module.ArchA_CanonicalResNet(9,[5]*8)
  self.register_buffer('reference',torch.as_tensor(reference,dtype=torch.float32).clone())
  width=128;self.node_encoder=nn.Sequential(nn.Linear(68,64),nn.GELU(),nn.Linear(64,width),nn.GELU())
  self.global_raw=nn.Sequential(nn.Linear(2304,width),nn.GELU(),nn.Linear(width,width));self.project=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,width))
  self.identity=nn.Parameter(torch.randn(1,8,width)*.02);self.pose_projections=nn.ModuleList([nn.Linear(5,width) for _ in range(8)])
  self.node_attention=nn.MultiheadAttention(width,4,batch_first=True);self.module_attention=nn.MultiheadAttention(width,4,batch_first=True)
  self.norm1=nn.LayerNorm(width);self.norm2=nn.LayerNorm(width);self.norm3=nn.LayerNorm(width)
  self.ff=nn.Sequential(nn.Linear(width,256),nn.GELU(),nn.Linear(256,width));self.heads=nn.ModuleList([nn.Linear(width,5) for _ in range(8)])
  for h in self.heads:nn.init.zeros_(h.weight);nn.init.zeros_(h.bias)
  for h in self.base.head.sub_heads:nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
  y,x=torch.meshgrid((torch.arange(16)+.5)/8-1,(torch.arange(16)+.5)/8-1,indexing='ij')
  xy=torch.stack((x.flatten(),y.flatten()),1).repeat(9,1);angles=torch.tensor(FIELDS/.132,dtype=torch.float32).repeat_interleave(256,0)
  self.register_buffer('coordinates',torch.cat((xy,angles),1))
 def forward(self,images):
  if tuple(images.shape[1:])!=(9,128,128):raise ValueError('Noisy9x128x128 ADU only')
  raw=(images-self.reference)/1024.;cnn=torch.asinh(raw)
  feat=self.base.pool(self.base.stage3(self.base.stage2(self.base.stage1(self.base.stem(cnn))))).flatten(1);base=self.base.head(feat)
  # Permute explicitly: each node contains the contiguous8x8 sensor patch.
  patches=raw.unfold(2,8,8).unfold(3,8,8).contiguous().reshape(len(raw),2304,64)
  nodes=self.node_encoder(torch.cat((patches,self.coordinates[None].expand(len(raw),-1,-1)),2))
  context=torch.stack([p(base[:,5*i:5*i+5]) for i,p in enumerate(self.pose_projections)],1)+self.identity+self.project(feat)[:,None]+self.global_raw(patches.mean(-1))[:,None]
  context=context+self.node_attention(self.norm1(context),nodes,nodes,need_weights=False)[0]
  n=self.norm2(context);context=context+self.module_attention(n,n,n,need_weights=False)[0];context=context+self.ff(self.norm3(context))
  return base+torch.cat([h(context[:,i]) for i,h in enumerate(self.heads)],1)
 def command(self,images):return -torch.tanh(self(images))
def restore(path,device='cuda'):
 p=torch.load(path,map_location='cpu',weights_only=False);m=JointNetwork(p['reference']);m.load_state_dict(p['model']);return m.to(device).eval(),p
