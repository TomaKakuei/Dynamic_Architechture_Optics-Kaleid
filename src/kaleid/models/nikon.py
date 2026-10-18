"""Nikon joint one-shot model."""
import torch
from .backbone import SpatialEncoder


class NikonModel(torch.nn.Module):
    """Native-node and module attention for seven Nikon bodies."""
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Module()
        self.base.networks = torch.nn.ModuleList(
            [SpatialEncoder(3, [5]*n) for n in (3, 3, 1)])
        self.register_buffer('raw_mean', torch.zeros(3, 49))
        self.register_buffer('raw_std', torch.ones(3, 49))
        self.register_buffer('target_mean', torch.zeros(35))
        self.register_buffer('target_std', torch.ones(35))
        self.register_buffer('node_coordinates', torch.zeros(147, 4))
        width=128
        self.node_encoder=torch.nn.Sequential(torch.nn.Linear(5,64),torch.nn.GELU(),torch.nn.Linear(64,width),torch.nn.GELU())
        self.raw_global=torch.nn.Sequential(torch.nn.Linear(147,width),torch.nn.GELU(),torch.nn.Linear(width,width))
        self.feature_projections=torch.nn.ModuleList([torch.nn.Linear(256,width) for _ in range(3)])
        self.pose_projections=torch.nn.ModuleList([torch.nn.Linear(n,width) for n in (15,15,5)])
        self.module_identity=torch.nn.Parameter(torch.zeros(1,3,width))
        self.cross_norm=torch.nn.LayerNorm(width)
        self.node_attention=torch.nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.module_norm=torch.nn.LayerNorm(width)
        self.module_attention=torch.nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.ff_norm=torch.nn.LayerNorm(width)
        self.ff=torch.nn.Sequential(torch.nn.Linear(width,256),torch.nn.GELU(),torch.nn.Linear(256,width))
        self.correction_heads=torch.nn.ModuleList([torch.nn.Linear(width,n) for n in (15,15,5)])
        for head in self.correction_heads:
            torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)

    def forward(self,image,raw):
        features=[];estimates=[]
        for network in self.base.networks:
            f=network.stage3(network.stage2(network.stage1(network.stem(image))))
            f=network.pool(f).flatten(1)
            features.append(f);estimates.append(network.head(f))
        native=((raw-self.raw_mean)/self.raw_std).flatten(1)
        coordinates=self.node_coordinates[None].expand(len(image),-1,-1)
        nodes=self.node_encoder(torch.cat((native[:,:,None],coordinates),dim=-1))
        context=torch.stack([fp(f)+pp(p) for fp,pp,f,p in zip(self.feature_projections,self.pose_projections,features,estimates)],dim=1)
        context=context+self.raw_global(native)[:,None]+self.module_identity
        context=context+self.node_attention(self.cross_norm(context),nodes,nodes,need_weights=False)[0]
        normalized=self.module_norm(context)
        context=context+self.module_attention(normalized,normalized,normalized,need_weights=False)[0]
        context=context+self.ff(self.ff_norm(context))
        correction=torch.cat([head(context[:,i]) for i,head in enumerate(self.correction_heads)],dim=-1)
        return torch.cat(estimates,dim=-1)+correction

    def physical_prediction(self,image,raw):
        return self(image,raw)*self.target_std+self.target_mean
