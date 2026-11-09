"""48-case refinement-input ablations and one lower-ridge diagnostic.
Same frozen prior and saved native4 estimate; no network retraining/prediction.
"""
from pathlib import Path
import json
import candidate as c
run,np,torch=c.run,c.np,c.torch
from trajectory_backward import TrajectoryStep
HERE=c.HERE


class Subset:
    def __init__(self,opt,columns):self.opt,self.columns=opt,columns
    def difference(self,x):return self.opt.difference(x)[:,self.columns]
    def response(self,x):return self.opt.response(x)[:,self.columns,:]


def low_ridge_step(x,prior,measured,opt,ridge=1e-6):
    residual=opt.difference(x)-measured;j=opt.response(x);jt=j.transpose(1,2)
    eye=torch.eye(35,dtype=x.dtype,device=x.device)
    gram=jt@j+ridge*eye;rhs=(jt@residual[:,:,None]).squeeze(-1)+ridge*(x-prior)
    delta=-torch.linalg.solve(gram,rhs[:,:,None]).squeeze(-1)
    delta=delta/torch.clamp(delta.abs().amax(1,keepdim=True)/.25,min=1.)
    old=residual.square().sum(1)+ridge*(x-prior).square().sum(1)
    out=x.clone();accepted=torch.zeros(len(x),device=x.device,dtype=torch.bool)
    for gain in (1.,.5,.25):
        ix=torch.nonzero(~accepted,as_tuple=True)[0]
        if not len(ix):break
        t=(x[ix]+gain*delta[ix]).clamp(-1.5,1.5)
        merit=(opt.difference(t)-measured[ix]).square().sum(1)+ridge*(t-prior[ix]).square().sum(1)
        good=merit<old[ix];out[ix[good]]=t[good];accepted[ix[good]]=True
    return out


def main():
    if (HERE/'ablation.json').exists():print('Already completed');return
    tick=run.exp.active_time();system=run.old.make_system('nikon')
    data=run.old.load_data(system);extra=np.load(run.HERE/'data/extra.npy',mmap_mode='r')
    with np.load(HERE/'values.npz') as z:d={k:z[k].copy() for k in z.files}
    lookup={int(v):i for i,v in enumerate(data['ids'])};rows=np.asarray([lookup[int(v)] for v in d['ids']])
    opt=c.sc.AllSensors(system);native=c.sc.NativeSensors(system)
    q=torch.as_tensor(d['q'],device='cuda');prior=torch.as_tensor(d['prior'],device='cuda');x0=torch.as_tensor(d['estimate4'],device='cuda')
    raw=torch.as_tensor(data['raw'][rows],device='cuda').double().flatten(1)/7.
    ex=torch.as_tensor(np.array(extra[rows],copy=True),device='cuda').double()
    measured=torch.cat((raw,ex[:,:197]/np.sqrt(197),ex[:,197:]/7.),1)
    report=dict(scope='Refinement inputs only; neural prior always uses existing540 values.48seven-body validation cases.',routes={})
    arrays={}
    def score(label,x):
        f,v,_=system.metrics(q-x,q);p=v&(f.max(1).values<.07)
        report['routes'][label]=dict(passes=int(p.sum()),count=48,invalid=int((~v).sum()),
            fields_mean=f.mean(0).cpu().tolist(),worst_p90=float(torch.quantile(f.max(1).values,.9)))
        arrays[label+'_estimate']=x.cpu().numpy();arrays[label+'_fields']=f.cpu().numpy()
    with torch.no_grad():
        for label,columns in [('add_centerABS',np.arange(344)),
            ('add_diff_0p1',np.r_[0:147,344:442]),('add_diff_0p1_0p15',np.r_[0:147,344:540]),
            ('add_centerABS_diff_0p1',np.arange(442))]:
            subset=Subset(opt,columns);obs=measured[:,columns];x=x0.clone()
            for _ in range(4):x=TrajectoryStep.apply(x,prior,obs,subset)
            score(label,x)
        for label,obs in [('native_lower_ridge_noisy',raw),('native_lower_ridge_clean',native.difference(q))]:
            x=x0.clone()
            for i in range(12):
                x=low_ridge_step(x,prior,obs,native)
                if i in (3,11):score(label+f'_total{i+5}',x)
    report['lower_ridge']=1e-6;report['original_ridge']=1e-4
    report['active_seconds']=run.exp.active_time()-tick
    (HERE/'ablation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez(HERE/'ablation_values.npz',ids=d['ids'],**arrays)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
