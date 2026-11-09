from optics import *
from sampling import sample
from geometry import Geometry
def scalar_reference(q,xy,fi):
 from rayoptics.elem.profiles import Spherical
 from rayoptics.raytr.raytrace import bend,reflect
 v=np.asarray(q).reshape(8,5)*SCALES;rots=[rotation(x[3:]) for x in v];centers=PIVOTS+v[:,:3]
 p=np.r_[np.asarray(xy)*SPEC['pupil_radius_mm'],0.];a=SPEC['angles_rad'][fi];d=np.array([0,np.sin(a),np.cos(a)]);op=float(p@d);ni=1.
 for i,s in enumerate(SURF[1:],1):
  gi=next((j for j,(a,b) in enumerate(GROUPS) if a<=i<=b),None);r=np.eye(3) if gi is None else rots[gi]
  origin=np.array([0,0,s['z']]) if gi is None else centers[gi]+r@np.array([0,0,s['z']-PIVOTS[gi,2]])
  pl=r.T@(p-origin);dl=r.T@d;profile=Spherical(c=1/s['r'] if s['r'] else 0.)
  t,h=profile.intersect(pl,dl,1e-12,s['zdir_in']);op+=ni*t;normal=profile.normal(h)
  if s['mat']=='mirror':out=reflect(dl,normal)
  else:
   no=1. if s['mat']=='air' else SPEC['n_silica'];out=bend(dl,normal,ni,no);ni=no
  p=r@h+origin;d=r@out
 return p,d,op
def numpy_metric(q,grid=65):
 xy=disk(grid);u,v=np.meshgrid(np.linspace(-1,1,grid),np.linspace(-1,1,grid));mask=(u*u+v*v<=1)&(u*u+v*v>=.05**2);A=np.c_[np.ones(len(xy)),xy];values=[];wrms=[]
 idx=2000*WAVE*(grid-1)/(2*SPEC['measured_na']);n=int(idx);f=idx-n
 for fi in range(9):
  phi,valid=numpy_phase(xy,fi,q);res=phi-A@np.linalg.lstsq(A[valid],phi[valid],rcond=None)[0]
  pupil=np.zeros((grid,grid),complex);pupil[mask]=np.exp(2j*np.pi*res)*valid
  psf=abs(np.fft.fft2(pupil,s=(2*grid,2*grid)))**2;otf=np.fft.ifft2(psf);value=abs((1-f)*otf[n,0]+f*otf[n+1,0])/otf[0,0].real
  values.append(float(value));wrms.append(float(np.sqrt(np.mean(res[valid]**2))))
 return dict(score=float(np.mean(values)-.3*np.std(values)),fields=values,wrms=wrms)
def run():
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;start=clock()
 geo=Geometry();o=Optics(grid=65,device='cuda');rows=[];maxray=0.;rng=np.random.default_rng(1933)
 for case in range(6):
  q=np.zeros(40) if not case else sample(case%3,case+100)[0]
  for fi in (0,4,8):
   xy=disk(17)[rng.choice(len(disk(17)),5,replace=False)];p,d,op,valid=numpy_trace(xy,SPEC['angles_rad'][fi],q)
   for j in range(len(xy)):
    pp,dd,oo=scalar_reference(q,xy[j],fi);maxray=max(maxray,float(abs(pp-p[j]).max()),float(abs(dd-d[j]).max()),abs(oo-op[j]))
  with torch.no_grad():r=o.metrics(torch.tensor(q[None],dtype=torch.float64,device='cuda'))
  ref=numpy_metric(q);rows.append(dict(case=case,score_error=abs(r['score'].item()-ref['score']),wrms_max_error=float(abs(r['wrms'][0].cpu().numpy()-ref['wrms']).max())))
 assert maxray<2e-8 and max(r['score_error'] for r in rows)<1e-7 and max(r['wrms_max_error'] for r in rows)<1e-6
 # Gradients through all 40 DOFs and the complete physical chain.
 small=Optics(grid=25,device='cuda');q=torch.tensor(sample(1,725)[0][None]*.01,dtype=torch.float64,device='cuda',requires_grad=True)
 m=small.metrics(q);loss=torch.log1p((m['wrms']/.07)**2).mean();grad=torch.autograd.grad(loss,q)[0][0].cpu().numpy();checks=[];h=1e-5
 for j in range(40):
  qp=q.detach().clone();qm=q.detach().clone();qp[0,j]+=h;qm[0,j]-=h
  with torch.no_grad():vp=torch.log1p((small.metrics(qp)['wrms']/.07)**2).mean().item();vm=torch.log1p((small.metrics(qm)['wrms']/.07)**2).mean().item()
  fd=(vp-vm)/(2*h);checks.append(dict(axis=j,autograd=float(grad[j]),finite_difference=fd,relative_error=abs(fd-grad[j])/max(abs(fd),abs(grad[j]),1e-3)))
 assert max(r['relative_error'] for r in checks)<.003,checks
 qs=np.array([sample(i%3,1000+i)[0] for i in range(192)]);values=[];support=[];waves=[]
 with torch.no_grad():
  for i in range(0,len(qs),4):
   m=o.metrics(torch.tensor(qs[i:i+4],dtype=torch.float64,device='cuda'));values.extend(m['score'].cpu().tolist());support.extend(m['support'].min(1).values.cpu().tolist());waves.extend(m['wrms'].mean(1).cpu().tolist())
  nominal=o.metrics(torch.zeros((1,40),dtype=torch.float64,device='cuda'))
 # Correct inverse labels restore the registered nominal exactly; no clipping.
 assert all(geo.path(np.zeros(40),q) for q in qs)
 fine=Optics(grid=129,device='cuda');dense=[]
 with torch.no_grad():
  for scale in (0.,.001,.003,.01,.03):
   qv=torch.tensor(qs[:6]*scale,dtype=torch.float64,device='cuda');a=o.metrics(qv)['score'];b=fine.metrics(qv)['score'];dense.extend(abs(a-b).cpu().tolist())
 assert max(dense)<.015,max(dense)
 result=dict(status='pass',seconds=clock()-start,scalar_rayoptics_max_error=maxray,independent_numpy_metrics=rows,all40_gradient_checks=checks,nominal={k:v.cpu().tolist() for k,v in nominal.items()},initial_distribution=dict(count=len(qs),passes=int(np.sum(np.array(values)>=.45)),q_mean=float(np.mean(values)),q_p90=float(np.quantile(values,.9)),wrms_mean=float(np.mean(waves)),min_support=min(support)),near_threshold_grid65_vs129_max_error=max(dense),geometry_nominal_min_clearance_mm=float(geo.clearances.min()),field_chief_span_mm=SPEC['chief_plane_span_mm'])
 atomic(HERE/'optics_check.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':run()
