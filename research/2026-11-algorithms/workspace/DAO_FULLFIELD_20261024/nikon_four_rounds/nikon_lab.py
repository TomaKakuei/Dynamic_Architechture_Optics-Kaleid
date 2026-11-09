"""Namespaced small-experiment helpers; avoids archived common.py collisions."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from experiment import System, atomic, active_time
import numpy as np
import torch

def tensor(q):return torch.as_tensor(q,dtype=torch.float64,device='cuda')

class Lab:
    def __init__(self):
        self.s=System('nikon');self.nk=self.s.legacy.nikon35
        self.native=self.s.legacy.NikonTensorOptics(nodes=self.nk.NODES,fields=self.nk.base.corrected.CONTROL_FIELDS_MM,device='cuda')
        with np.load(HERE.parent/'nikon_debug_20261024/small_sample_endpoints.npz') as z:
            self.data={k:z[k].copy() for k in z.files}
        self.devix=np.r_[96:104,128:136,160:168]
        self.confirmix=np.r_[104:112,136:144,168:176]
        self.q0=self.data['large_fullfield'][self.devix]
        self.hidden=self.data['q'][self.devix]
        self.seeds=self.data['ids'][self.devix]
        self.ref=self.observation(np.zeros((1,35)))[0]

    def residual_tensor(self,q):
        m=self.s.optics.metrics(q)
        return torch.cat((m['absolute'][:,:1],m['difference'][:,1:]),dim=1)

    def fields(self,q):
        with torch.no_grad():r=self.residual_tensor(tensor(q))
        return torch.sqrt(r.square().mean(-1)).cpu().numpy()

    def score(self,q):
        w=self.fields(q)
        with torch.no_grad():valid=self.s.optics.metrics(tensor(q))['valid'].cpu().numpy()
        return dict(count=len(w),fullfield_pass=int(((w.max(1)<.07)&valid).sum()),
                    center_pass=int(((w[:,0]<.07)&valid).sum()),invalid=int((~valid).sum()),
                    field_mean=w.mean(0).tolist(),worst_mean=float(w.max(1).mean()),fields=w.tolist())

    def observation(self,q,kind='native'):
        with torch.no_grad():
            t=tensor(q);m=self.s.optics.metrics(t)
            if kind=='audit':return self.residual_tensor(t).flatten(1).cpu().numpy()/np.sqrt(m['absolute'].shape[-1])
            n=self.native.metrics(t)
            return torch.cat((m['absolute'][:,0]/np.sqrt(m['absolute'].shape[-1]),
                              n['difference'].flatten(1)/np.sqrt(49)),dim=1).cpu().numpy()

    def jacobian(self,q,kind='native',step=1e-4):
        q=np.asarray(q);eye=np.eye(35)*step
        points=np.concatenate((q[:,None,:]+eye,q[:,None,:]-eye),axis=1).reshape(-1,35)
        values=np.concatenate([self.observation(points[i:i+48],kind) for i in range(0,len(points),48)])
        v=values.reshape(len(q),70,-1)
        return ((v[:,:35]-v[:,35:])/(2*step)).transpose(0,2,1)

    def noise(self,seeds,iteration,count=None):
        # Exact historical per-ray0.015waves, with the same field RMS scaling.
        nc=len(self.nk.AUDIT_NODES);n=nc+147
        scale=np.r_[np.full(nc,.015/np.sqrt(nc)),np.full(147,.015/7)]
        return np.stack([np.random.RandomState((int(seed)+7919*(iteration+1))%(2**32)).normal(size=n)*scale for seed in seeds])

def inverse(j,lam=1e-4):
    u,s,v=np.linalg.svd(j,full_matrices=False)
    return (v.swapaxes(-1,-2)*(s/(s*s+lam))[...,None,:])@u.swapaxes(-1,-2)

def cap(step,limit=.25):return step/np.maximum(1,np.max(abs(step),axis=-1,keepdims=True)/limit)

def brief(v):return {k:x for k,x in v.items() if k!='fields'}
