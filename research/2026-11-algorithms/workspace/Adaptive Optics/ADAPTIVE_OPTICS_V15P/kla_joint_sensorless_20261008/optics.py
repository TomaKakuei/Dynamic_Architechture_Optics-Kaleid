"""Global rigid poses, physical angular launches and fixed image reference.

Scalar pupil diffraction model at 193.3 nm. The Mangin section is fixed,
including the registered 5% entrance obscuration approximation. Apertures
are fixed before sampling, never resized around perturbed rays.
"""
from common import *
import torch
from torch import nn
SURF=SPEC['surfaces'];Z=np.array([s['z'] for s in SURF]);PIVOTS=np.array([[0,0,(Z[a]+Z[b])/2] for a,b in GROUPS])
def numpy_trace(xy,angle,q=None,prefix_only=False,apertures=True,history=False):
 q=np.zeros(40) if q is None else np.asarray(q);v=q.reshape(8,5)*SCALES
 rotations=np.array([rotation(t[3:]) for t in v]);centers=PIVOTS+v[:,:3]
 p=np.c_[np.asarray(xy)*SPEC['pupil_radius_mm'],np.zeros(len(xy))];d=np.tile([0,np.sin(angle),np.cos(angle)],(len(xy),1))
 op=(p*d).sum(1);valid=np.ones(len(p),bool);ni=1.;hits=[]
 for i,s in enumerate(SURF[1:],1):
  if prefix_only and i>=15:break
  gi=next((j for j,(a,b) in enumerate(GROUPS) if a<=i<=b),None)
  r=np.eye(3) if gi is None else rotations[gi];origin=np.array([0,0,s['z']]) if gi is None else centers[gi]+r[:,2]*(s['z']-PIVOTS[gi,2])
  pl=(p-origin)@r;dl=d@r;cv=1/s['r'] if s['r'] else 0.
  shift=-(pl*dl).sum(1);pp=pl+shift[:,None]*dl;c=cv*(pp*pp).sum(1)-2*pp[:,2];b=cv*(pp*dl).sum(1)-dl[:,2];D=b*b-cv*c
  valid&=D>0;direction=s['zdir_in'];t=shift+c/(direction*np.sqrt(np.maximum(D,1e-30))-b)
  h=pl+t[:,None]*dl;n=-cv*h+[0,0,1];n/=np.linalg.norm(n,axis=1)[:,None]
  if apertures:valid&=np.linalg.norm(h[:,:2],axis=1)<=s['aperture_mm']+1e-9
  valid&=(t>=-1e-8)&np.isfinite(t);op+=ni*t;cos=(dl*n).sum(1)
  if s['mat']=='mirror':out=dl-2*cos[:,None]*n
  else:
   no=1. if s['mat']=='air' else SPEC['n_silica'];rad=no*no-ni*ni*(1-cos*cos);valid&=rad>=0
   out=(ni*dl+(np.copysign(np.sqrt(np.maximum(rad,1e-30)),cos)-ni*cos)[:,None]*n)/no;ni=no
  p=h@r.T+origin;d=out@r.T
  if history:hits.append(h.copy())
 result=(p,d,op,valid)
 return (*result,hits) if history else result

def numpy_phase(xy,fi,q):
 p,d,op,valid=numpy_trace(xy,SPEC['angles_rad'][fi],q)
 focus=np.asarray(SPEC['reference_points_mm'][fi]);phase=(op+np.linalg.norm(focus-p,axis=1))/WAVE
 return phase-phase.mean(),valid

class Optics(nn.Module):
 def __init__(self,grid=65,xy=None,fields=None,device='cpu'):
  super().__init__();self.fields=list(range(9)) if fields is None else list(fields);self.grid=grid
  if xy is None:
   line=np.linspace(-1,1,grid);x,y=np.meshgrid(line,line);mask=(x*x+y*y<=1)&(x*x+y*y>=.05**2);xy=np.c_[x[mask],y[mask]]
   self.register_buffer('disk',torch.tensor(mask,device=device))
  xy=np.asarray(xy)
  def reg(k,v):self.register_buffer(k,torch.tensor(v,dtype=torch.float64,device=device))
  reg('xy',xy);reg('design',np.c_[np.ones(len(xy)),xy]);reg('scale',SCALES);reg('pivots',PIVOTS)
  launches=[numpy_trace(xy,SPEC['angles_rad'][i],prefix_only=True) for i in self.fields]
  for k,j in [('p0',0),('d0',1),('o0',2)]:reg(k,np.stack([t[j] for t in launches]))
  self.register_buffer('v0',torch.tensor(np.stack([t[3] for t in launches]),device=device))
  reg('reference',np.asarray(SPEC['reference_points_mm'])[self.fields])
  self.freq_index=2000*WAVE*(grid-1)/(2*SPEC['measured_na'])
 def poses(self,q):
  s=q.to(torch.float64).reshape(-1,8,5)*self.scale
  tx,ty=torch.deg2rad(s[:,:,3]),torch.deg2rad(s[:,:,4]);cx,sx,cy,sy=torch.cos(tx),torch.sin(tx),torch.cos(ty),torch.sin(ty);z=torch.zeros_like(tx)
  r=torch.stack([cy,sy*sx,sy*cx,z,cx,-sx,-sy,cy*sx,cy*cx],-1).reshape(-1,8,3,3)
  return self.pivots+s[:,:,:3],r
 def trace(self,q):
  center,rots=self.poses(q);p=self.p0.expand(len(q),-1,-1,-1);d=self.d0.expand_as(p);op=self.o0.expand(len(q),-1,-1);valid=self.v0.expand(len(q),-1,-1);ni=1.
  for i in range(15,len(SURF)):
   s=SURF[i];gi=(i-15)//2;r=rots[:,gi];origin=center[:,gi]+r[:,:,2]*(s['z']-PIVOTS[gi,2])
   pl=torch.einsum('bji,bfnj->bfni',r,p-origin[:,None,None]);dl=torch.einsum('bji,bfnj->bfni',r,d)
   cv=1/s['r'];shift=-(pl*dl).sum(-1);pp=pl+shift[...,None]*dl;c=cv*pp.square().sum(-1)-2*pp[...,2];b=cv*(pp*dl).sum(-1)-dl[...,2];D=b*b-cv*c
   valid=valid&(D>0);t=shift+c/(torch.sqrt(D.clamp_min(1e-30))-b);h=pl+t[...,None]*dl
   n=-cv*h+h.new_tensor([0.,0.,1.]);n=n/torch.linalg.vector_norm(n,dim=-1,keepdim=True)
   valid=valid&(h[...,:2].square().sum(-1)<=(s['aperture_mm']+1e-9)**2)&(t>=-1e-8)
   op=op+ni*t;cos=(dl*n).sum(-1);no=1. if s['mat']=='air' else SPEC['n_silica'];rad=no*no-ni*ni*(1-cos*cos);valid=valid&(rad>=0)
   out=(ni*dl+(torch.copysign(torch.sqrt(rad.clamp_min(1e-30)),cos)-ni*cos)[...,None]*n)/no;ni=no
   p=torch.einsum('bij,bfnj->bfni',r,h)+origin[:,None,None];d=torch.einsum('bij,bfnj->bfni',r,out)
  return p,d,op,valid
 def phase(self,q):
  p,d,op,valid=self.trace(q);phi=(op+torch.linalg.vector_norm(self.reference[None,:,None]-p,dim=-1))/WAVE
  phi=phi-phi.mean(-1,keepdim=True)
  return phi,valid&torch.isfinite(phi)
 def residual(self,q):
  phi,valid=self.phase(q);w=valid.double();A=self.design
  gram=torch.einsum('nr,bfn,ns->bfrs',A,w,A);rhs=torch.einsum('nr,bfn->bfr',A,w*phi)
  c=torch.linalg.solve(gram+torch.eye(3,dtype=torch.float64,device=q.device)*1e-12,rhs[...,None])[...,0]
  res=phi-torch.einsum('nr,bfr->bfn',A,c)
  return res,valid
 def metrics(self,q):
  phase,valid=self.residual(q);w=valid.double();wrms=torch.sqrt((phase.square()*w).sum(-1)/w.sum(-1).clamp_min(1)+1e-20)
  pupil=torch.zeros((*phase.shape[:2],self.grid,self.grid),device=q.device,dtype=torch.complex128)
  pupil[:,:,self.disk]=torch.exp(2j*torch.pi*phase)*w
  # Pupil autocorrelation, with physical 2000 lp/mm interpolated on the OTF axis.
  ft=torch.fft.fft2(pupil,s=(2*self.grid,2*self.grid));otf=torch.fft.ifft2(ft.abs().square())
  n=int(self.freq_index);f=self.freq_index-n;den=otf[:,:,0,0].real.clamp_min(1e-30)
  a=(1-f)*otf[:,:,n,0]+f*otf[:,:,n+1,0];mtf=a.abs()/den
  mean=mtf.mean(-1);std=torch.sqrt((mtf-mean[:,None]).square().mean(-1)+1e-20);score=mean-.3*std
  return dict(score=score,fields=mtf,wrms=wrms,support=w.mean(-1),valid=(w.mean(-1)>=.98).all(-1)&torch.isfinite(score))
