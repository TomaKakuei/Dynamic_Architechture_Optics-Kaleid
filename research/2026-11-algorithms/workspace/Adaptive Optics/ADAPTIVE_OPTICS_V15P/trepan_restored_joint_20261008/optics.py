"""Fixed-prefix, globally posed, explicit-OPL ray trace. Float64 throughout.

The fixed relay is cached at S66. Eight objective bodies move independently;
the terminal image is a reference only. No RayOptics finite-object waveabr.
"""
from common import *
from geometry import Geometry,SURF
import torch
from torch import nn
def numpy_trace(xy,field,q=None,prefix_only=False):
 q=np.zeros(40) if q is None else np.asarray(q);geo=Geometry();centers,rots=geo.poses(q)
 p=np.c_[xy*3,np.zeros(len(xy))];d=np.tile(np.r_[np.tan(np.deg2rad(field)),1.],(len(xy),1));d/=np.linalg.norm(d,axis=1)[:,None]
 opl=(p*d).sum(1);valid=np.ones(len(xy),bool)
 for s in SPEC['surface']:
  if prefix_only and s['sid']>=68:break
  gi=next((i for i,(a,b) in enumerate(GROUPS) if a<=s['sid']<=b),None)
  rot=np.eye(3) if gi is None else rots[gi]
  origin=np.array([0,0,s['z']]) if gi is None else centers[gi]+rot@np.array([0,0,s['z']-geo.pivots[gi,2]])
  bp=(p-origin)@rot;bd=d@rot
  if np.isinf(s['r']):t=-bp[:,2]/bd[:,2];hit=bp+t[:,None]*bd;n=np.tile([0.,0.,1.],(len(p),1))
  else:
   cv=1/s['r'];perp=-(bp*bd).sum(1);pp=bp+perp[:,None]*bd
   c=cv*(pp*pp).sum(1)-2*pp[:,2];beta=cv*(bd*pp).sum(1)-bd[:,2];disc=beta*beta-cv*c
   valid&=disc>0;t=perp+c/(np.sqrt(np.maximum(disc,1e-30))-beta);hit=bp+t[:,None]*bd;n=-cv*hit+[0,0,1];n/=np.linalg.norm(n,axis=1)[:,None]
  valid&=np.linalg.norm(hit[:,:2],axis=1)<=s['sd']+1e-9
  opl+=s['ni']*t;cos=(bd*n).sum(1);rad=s['no']**2-s['ni']**2*(1-cos*cos);valid&=rad>=0
  dd=(s['ni']*bd+(np.copysign(np.sqrt(np.maximum(rad,1e-30)),cos)-s['ni']*cos)[:,None]*n)/s['no']
  p=hit@rot.T+origin;d=dd@rot.T
 return p,d,opl,valid
def references():
 refs=[];focus=SPEC['image_z']+SPEC['focus_offset_mm']
 for f in FIELDS:
  p,d,_,_=numpy_trace(np.zeros((1,2)),f);c=p[0]+(focus-p[0,2])/d[0,2]*d[0]
  z=focus-263.157+np.sqrt(263.157**2-np.sum(c[:2]**2))
  refs.append(p[0]+(z-p[0,2])/d[0,2]*d[0])
 return np.asarray(refs)
class Optics(nn.Module):
 def __init__(self,grid=33,xy=None,fields=None,device='cpu'):
  super().__init__();self.fields=list(range(9)) if fields is None else list(fields)
  xy=disk(grid) if xy is None else np.asarray(xy);self.grid=grid
  self.surfaces=[s for s in SPEC['surface'] if s['sid']>=68]
  def reg(k,v):self.register_buffer(k,torch.tensor(v,dtype=torch.float64,device=device))
  reg('xy',xy);reg('design',np.c_[np.ones(len(xy)),xy,(xy*xy).sum(1)])
  launches=[numpy_trace(xy,FIELDS[i],prefix_only=True) for i in self.fields]
  for k,ix in [('p0',0),('d0',1),('o0',2)]:reg(k,np.stack([x[ix] for x in launches]))
  self.register_buffer('v0',torch.tensor(np.stack([x[3] for x in launches]),device=device))
  reg('centers',references()[self.fields]);reg('pivots',Geometry().pivots);reg('scale',SCALES)
 def poses(self,q):
  s=q.reshape(-1,8,5)*self.scale
  tx,ty=torch.deg2rad(s[:,:,3]),torch.deg2rad(s[:,:,4]);cx,sx,cy,sy=torch.cos(tx),torch.sin(tx),torch.cos(ty),torch.sin(ty);z=torch.zeros_like(tx)
  r=torch.stack([cy,sy*sx,sy*cx,z,cx,-sx,-sy,cy*sx,cy*cx],-1).reshape(-1,8,3,3)
  return self.pivots+s[:,:,:3],r
 def trace(self,q):
  center,rots=self.poses(q);p=self.p0.expand(len(q),-1,-1,-1);d=self.d0.expand_as(p);op=self.o0.expand(len(q),-1,-1);valid=self.v0.expand(len(q),-1,-1)
  for s in self.surfaces:
   gi=next(i for i,(a,b) in enumerate(GROUPS) if a<=s['sid']<=b);r=rots[:,gi];origin=center[:,gi]+r[:,:,2]*(s['z']-self.pivots[gi,2])
   bp=torch.einsum('bji,bfnj->bfni',r,p-origin[:,None,None]);bd=torch.einsum('bji,bfnj->bfni',r,d)
   cv=0 if np.isinf(s['r']) else 1/s['r'];perp=-(bp*bd).sum(-1);pp=bp+perp[...,None]*bd
   c=cv*(pp*pp).sum(-1)-2*pp[...,2];beta=cv*(bd*pp).sum(-1)-bd[...,2];disc=beta*beta-cv*c
   valid=valid&(disc>0);t=perp+c/(torch.sqrt(disc.clamp_min(1e-30))-beta)
   hit=bp+t[...,None]*bd;n=-cv*hit+hit.new_tensor([0.,0.,1.]);n=n/torch.linalg.vector_norm(n,dim=-1,keepdim=True)
   valid=valid&(hit[...,:2].square().sum(-1)<=(s['sd']+1e-9)**2)
   op=op+s['ni']*t;cos=(bd*n).sum(-1);rad=s['no']**2-s['ni']**2*(1-cos*cos);valid=valid&(rad>=0)
   dd=(s['ni']*bd+(torch.copysign(torch.sqrt(rad.clamp_min(1e-30)),cos)-s['ni']*cos)[...,None]*n)/s['no']
   p=torch.einsum('bij,bfnj->bfni',r,hit)+origin[:,None,None];d=torch.einsum('bij,bfnj->bfni',r,dd)
  return p,d,op,valid
 def phase(self,q):
  p,d,op,valid=self.trace(q);delta=p-self.centers[None,:,None];b=(delta*d).sum(-1);disc=b*b-delta.square().sum(-1)+10000
  valid=valid&(disc>0);t=-b-torch.sqrt(disc.clamp_min(1e-30));phase=(op+SPEC['n_image']*t)/.000910
  phase=phase-phase.mean(-1,keepdim=True)
  return phase,valid
 def metrics(self,q):
  phase,valid=self.phase(q);w=valid.to(phase.dtype);A=self.design
  gram=torch.einsum('nr,bfn,ns->bfrs',A,w,A);rhs=torch.einsum('nr,bfn->bfr',A,w*phase)
  c=torch.linalg.solve(gram+torch.eye(4,dtype=q.dtype,device=q.device)*1e-12,rhs[...,None])[...,0]
  res=(phase-torch.einsum('nr,bfr->bfn',A,c))*w
  wrms=torch.sqrt(res.square().sum(-1)/w.sum(-1).clamp_min(1)+1e-20)
  return wrms,valid.double().mean(-1)
 def mechanical_loss(self,q):
  # Smooth gap lower bounds in global Z except nested S80/S81. Exact 3D path
  # certification remains mandatory for scored endpoints, independently.
  centers,rots=self.poses(q);g=Geometry();front=[];back=[]
  for i in range(8):
   for face,dest,isfront in [(g.faces[i][0],front,True),(g.faces[i][1],back,False)]:
    r,cap,outer,dz=face;edge=float(0 if np.isinf(r) else sag_local(r,cap));lo=min(dz,dz+edge);hi=max(dz,dz+edge)
    spread=outer*torch.linalg.vector_norm(rots[:,i,2,:2],dim=-1)
    dest.append(centers[:,i,2]+(lo if isfront else hi)*rots[:,i,2,2]+(-spread if isfront else spread))
  gaps=torch.stack([front[i+1]-back[i] for i in range(7)],1)
  sl,sr=SURF[80],SURF[81];cl=centers[:,5]+rots[:,5,:,2]*(sl['z']+sl['r']-self.pivots[5,2]);cr=centers[:,6]+rots[:,6,:,2]*(sr['z']+sr['r']-self.pivots[6,2]);dc=cr-cl;dr=sl['r']-sr['r']
  gaps[:,5]=dc[:,2]+torch.sqrt((dr*dr-dc[:,:2].square().sum(-1)).clamp_min(1e-12))
  return torch.relu((.002-gaps)/.1).square().sum(1)
def sag_local(r,x):return x*x/(r+np.copysign(np.sqrt(r*r-x*x),r))
