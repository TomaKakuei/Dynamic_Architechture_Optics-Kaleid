from optics import *
from rayoptics.elem.profiles import Spherical
from rayoptics.raytr.raytrace import bend
def scalar_reference(q,xy,field):
 geo=Geometry();centers,rots=geo.poses(q);p=np.r_[xy*3,0.];d=np.r_[np.tan(np.deg2rad(field)),1.];d/=np.linalg.norm(d);op=float(p@d)
 for s in SPEC['surface']:
  gi=next((i for i,(a,b) in enumerate(GROUPS) if a<=s['sid']<=b),None);r=np.eye(3) if gi is None else rots[gi]
  origin=np.array([0.,0.,s['z']]) if gi is None else centers[gi]+r@np.array([0.,0.,s['z']-geo.pivots[gi,2]])
  lp=r.T@(p-origin);ld=r.T@d;profile=Spherical(c=0 if np.isinf(s['r']) else 1/s['r'])
  distance,hit=profile.intersect(lp,ld,1e-12,1);out=bend(ld,profile.normal(hit),s['ni'],s['no']);op+=s['ni']*distance;p=r@hit+origin;d=r@out
 return p,d,op
def run():
 t=clock();g=Geometry();print('NOMINAL_GEOMETRY',g.bounds(np.zeros(40)),flush=True);assert g.path(np.zeros(40))
 m=Optics(grid=65,device='cuda');q=torch.zeros((1,40),dtype=torch.float64,device='cuda')
 w,s=m.metrics(q);print('NOMINAL',w.detach().cpu().numpy(),s.cpu().numpy(),flush=True)
 assert abs(float(w[0,0])-.035446)<.0004 and abs(float(w.mean())-.052283)<.0005 and float(s.min())==1
 rng=np.random.default_rng(108);states=[np.zeros(40)]+[rng.uniform(-.8,.8,40) for _ in range(8)]
 errors=[]
 for state in states:
  for fi in (0,2,4):
   xy=rng.uniform(-.5,.5,(4,2));v=numpy_trace(xy,FIELDS[fi],state)
   model=Optics(xy=xy,fields=[fi]);r=model.trace(torch.tensor(state[None],dtype=torch.float64))
   for j in range(4):
    a=scalar_reference(state,xy[j],FIELDS[fi]);errors.append([np.max(abs(a[k]-v[k][j])) for k in range(3)])
   assert max(float((r[k][0,0]-torch.tensor(v[k])).abs().max()) for k in range(3))<1e-8
 err=np.max(errors,axis=0);assert max(err)<2e-8;print('SCALAR_RAYOPTICS_ERROR',err,flush=True)
 center=Optics(grid=33,fields=[0],device='cuda');q=torch.tensor(rng.uniform(-.3,.3,(1,40)),dtype=torch.float64,device='cuda',requires_grad=True)
 value=center.metrics(q)[0].square().sum();grad=torch.autograd.grad(value,q)[0];check=[]
 for axis in [0,2,3,8,12,19,20,27,30,34,37,39]:
  h=1e-4;e=torch.zeros_like(q);e[0,axis]=h
  with torch.no_grad(): fd=float((center.metrics(q+e)[0].square().sum()-center.metrics(q-e)[0].square().sum())/(2*h))
  ad=float(grad[0,axis]);check.append(dict(axis=axis,ad=ad,fd=fd,error=abs(ad-fd)/max(abs(fd),1e-6)))
 assert max(x['error'] for x in check)<.005,check
 # Shared-gap shift regression: all eight independent Z changes must persist.
 qq=np.zeros(40);qq[2::5]=np.arange(8)*.04;c,_=g.poses(qq);assert np.allclose(c[:,2]-g.pivots[:,2],qq[2::5]*.1)
 # Forced overlap at the small nested gap must be rejected.
 collide=np.zeros(40);collide[27]=1;collide[32]=-1
 assert not g.state(collide) and not g.path(collide)
 result=dict(status='pass',nominal_wrms=w[0].cpu().tolist(),support=s[0].cpu().tolist(),rayoptics_cases=108,rayoptics_errors=err.tolist(),gradient=check,nominal_clearances=g.bounds(np.zeros(40)).tolist(),seconds=clock()-t)
 atomic(HERE/'optics_check.json',result);print('PASS',result['seconds'],flush=True)
if __name__=='__main__':run()
