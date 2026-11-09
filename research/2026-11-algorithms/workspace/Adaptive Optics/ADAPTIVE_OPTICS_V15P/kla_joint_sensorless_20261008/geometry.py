"""Conservative complete-path clearance of fixed glass envelopes.

Nominal graph separation is converted to Euclidean distance using a global
slope bound. Rigid-body displacement bounds then certify every point of
the straight actuator path. Rejection means uncertified, not proven collision.
No projection, shared-gap mutation or truth-dependent actuator clipping.
"""
from common import *
SURF=SPEC['surfaces'];MARGIN=.001
class Geometry:
 def __init__(self):
  self.pivots=np.array([[0,0,(SURF[a]['z']+SURF[b]['z'])/2] for a,b in GROUPS])
  self.outer=np.array([max(SURF[a]['aperture_mm'],SURF[b]['aperture_mm']) for a,b in GROUPS])
  self.extent=[]
  for i,(a,b) in enumerate(GROUPS):
   vals=[SURF[k]['z']-self.pivots[i,2]+sag(SURF[k]['r'],r) for k in (a,b) for r in (0,SURF[k]['aperture_mm'])]
   self.extent.append(float(np.hypot(self.outer[i],max(abs(v) for v in vals))))
  self.extent=np.array(self.extent);self.pairs=[];self.clearances=[]
  for i in range(8):
   for j in range(i+1,8):
    left=SURF[GROUPS[i][1]];right=SURF[GROUPS[j][0]];r=np.linspace(0,max(self.outer[i],self.outer[j]),20001)
    def face(s):return s['z']+sag(s['r'],np.minimum(r,s['aperture_mm']))
    slopes=[s['aperture_mm']/np.sqrt(s['r']**2-s['aperture_mm']**2) for s in (left,right)]
    gap=float((face(right)-face(left)).min()-(sum(slopes)*(r[1]-r[0])/2))
    # Same-XY graph separation / sqrt(1+L^2) bounds Euclidean distance.
    clearance=gap/np.sqrt(1+max(slopes)**2)
    if clearance<=MARGIN:raise ValueError(f'Nominal glass envelopes cannot be certified: {i},{j},{clearance}')
    self.pairs.append((i,j));self.clearances.append(clearance)
  self.clearances=np.asarray(self.clearances)
 def motion(self,q):
  v=np.asarray(q).reshape(8,5)*SCALES;ang=np.deg2rad(abs(v[:,3:]).sum(1))
  return np.linalg.norm(v[:,:3],axis=1)+self.extent*ang
 def bounds(self,q):
  move=self.motion(q);return self.clearances-np.array([move[i]+move[j] for i,j in self.pairs])
 def state(self,q):return bool(np.isfinite(q).all() and abs(np.asarray(q)).max()<=2.000001 and self.bounds(q).min()>=MARGIN)
 def path(self,end,start=None):
  start=np.zeros(40) if start is None else np.asarray(start)
  # Norm and absolute-angle bounds are convex along the actuator segment.
  a=self.motion(start);b=self.motion(end);worst=np.maximum(a,b)
  return bool(self.state(start) and self.state(end) and min(c-worst[i]-worst[j] for c,(i,j) in zip(self.clearances,self.pairs))>=MARGIN)
 def penalty(self,q):
  import torch
  v=q.reshape(-1,8,5)*q.new_tensor(SCALES);move=torch.linalg.vector_norm(v[:,:,:3],dim=-1)+q.new_tensor(self.extent)*torch.deg2rad(v[:,:,3:].abs().sum(-1))
  gaps=torch.stack([c-move[:,i]-move[:,j] for c,(i,j) in zip(self.clearances,self.pairs)],1)
  return torch.relu((.002-gaps)/.08).square().sum(1)
