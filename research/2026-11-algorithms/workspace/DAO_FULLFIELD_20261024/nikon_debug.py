"""Bounded diagnostic, no optimizer and no training/checkpoint mutation."""
from pathlib import Path
import json
import numpy as np
import torch
from experiment import System, HERE, ROOT, atomic, active_time

OUT = HERE/'nikon_debug_20261024'

def score(system, endpoint):
    with torch.no_grad():
        q = torch.as_tensor(endpoint, device='cuda', dtype=torch.float64)
        fields, valid, absolute = system.metrics(q, q)
    w = fields.cpu().numpy()
    good = valid.cpu().numpy()
    return dict(count=len(w), center_pass=int((good & (w[:,0]<.07)).sum()),
                fullfield_pass=int((good & (w.max(1)<.07)).sum()),
                field_mean=w.mean(0).tolist(), invalid=int((~good).sum()), fields=w.tolist())

def main():
    start=active_time(); OUT.mkdir(exist_ok=True)
    system=System('nikon'); nk=system.legacy.nikon35
    large=system.model('large').eval()
    large.load_state_dict(torch.load(HERE/'nikon_large/candidate.pt',map_location='cpu',weights_only=False)['model'])
    base=system.model('standard').eval()
    # First32 rows of each group's first original training/validation file.
    rows=[]
    sources=system.legacy.sources()
    for i in (0,100,200,40,140,240):
        path,group,training,_=sources[i]
        with np.load(path) as z:
            rows.append(dict(image=z['maps'][:32].astype(np.float32), raw=z['raw'][:32].astype(np.float32),
                             q=z['targets'][:32].astype(np.float64), ids=z['seeds'][:32],
                             train=np.full(32,training),cohort=np.full(32,group)))
    data={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    results={'no_training':True,'metric':'center ABS, both edges DIFF against fixed corrected bias; threshold0.07waves',
             'sample_rule':'First32 rows of first original train and validation file in each of3groups;192total',
             'sources':[str(sources[i][0]) for i in (0,100,200,40,140,240)],
             'nominal':score(system,np.zeros((1,35))), 'models':{}}
    endpoints={}
    for label,model in [('archived_center',base),('large_fullfield',large)]:
        with torch.no_grad():
            endpoint,initial,command,pose=system.forward(model,data,np.arange(len(data['q'])))
        e=endpoint.cpu().numpy();endpoints[label]=e
        results['models'][label]={split:score(system,e[data['train']==training]) for split,training in [('train',True),('validation',False)]}
        results['models'][label].update(pose_loss=float(pose),endpoint_rms=float(np.sqrt(np.mean(e*e))),
                                        clipped_command_coordinates=int((command.abs()>=1.499999).sum()))
    # Original11 historical endpoints are scored with correct edge DIFF.
    records=[]
    for seed in nk.TEST_SEEDS:
        prior=json.loads((ROOT/'oneshot_resources_11_20261014/cases'/f'nikon_{seed}_one_shot.json').read_text())
        hidden,_=nk.sample(seed)
        records.append((seed,hidden+np.asarray(prior['command']),prior['terminal']['center']))
    old11=score(system,np.asarray([r[1] for r in records]))
    old11['seeds']=[r[0] for r in records]
    old11['saved_center_max_difference']=float(np.max(np.abs(np.asarray(old11['fields'])[:,0]-[r[2] for r in records])))
    results['original11_archived_endpoints']=old11
    # Endpoint gradients isolate the physical objective; these are NOT network updates.
    q=torch.tensor(endpoints['large_fullfield'][96:104],device='cuda',dtype=torch.float64,requires_grad=True)
    w,_,_=system.metrics(q,q.detach())
    ell=torch.log1p((w/.07).square()); term=ell/(1+ell)
    bounded=(.5*term.mean(1)+.5*term.amax(1)).mean()
    gb=torch.autograd.grad(bounded,q,retain_graph=True)[0]
    pseudo=(torch.sqrt(1+(w/.07).square())-1).mean()
    gp=torch.autograd.grad(pseudo,q,retain_graph=True)[0]
    gc=torch.autograd.grad(term[:,0].mean(),q,retain_graph=True)[0]
    ge=torch.autograd.grad(term[:,1:].mean(),q)[0]
    direction=gb/gb.norm(); h=1e-4
    def loss_at(t):
        with torch.no_grad():
            f,_,_=system.metrics(t,t)
            l=torch.log1p((f/.07).square());v=l/(1+l)
            return (.5*v.mean(1)+.5*v.amax(1)).mean()
    fd=(loss_at(q.detach()+h*direction)-loss_at(q.detach()-h*direction))/(2*h)
    analytic=(gb*direction).sum()
    x=torch.tensor([.035,.07,.1,.5,1.,2.],dtype=torch.float64,requires_grad=True)
    ell=torch.log1p((x/.07).square()); b=ell/(1+ell)
    slope=torch.autograd.grad(b.sum(),x)[0]
    results['loss_diagnostic']=dict(fields=w.detach().cpu().tolist(),bounded_gradient_norm=float(gb.norm()),
        non_saturating_gradient_norm=float(gp.norm()),center_gradient_norm=float(gc.norm()),edge_gradient_norm=float(ge.norm()),
        center_edge_gradient_cosine=float(torch.nn.functional.cosine_similarity(gc.flatten(),ge.flatten(),dim=0)),
        finite_difference=float(fd),analytic=float(analytic),relative_error=float(abs(fd-analytic)/abs(analytic)),
        scalar_wrms=x.detach().tolist(),bounded_penalty_slope=slope.tolist())
    # Oracle only: uniform residual-pose contraction identifies required pose accuracy.
    results['oracle_residual_contraction']={str(scale):score(system,endpoints['large_fullfield'][96:]*scale)
                                            for scale in (1.,.5,.1,.03,0.)}
    results['elapsed_seconds']=active_time()-start
    atomic(OUT/'diagnostic.json',results)
    np.savez_compressed(OUT/'small_sample_endpoints.npz',**data,**endpoints)
    print(json.dumps({k:v for k,v in results.items() if k not in ('models','original11_archived_endpoints','oracle_residual_contraction')},indent=2))
    for label,res in results['models'].items():
        print(label,{k:{a:b for a,b in v.items() if a!='fields'} if isinstance(v,dict) else v for k,v in res.items()})
    print('old11', {k:v for k,v in old11.items() if k!='fields'})
    print('contraction',{k:{a:b for a,b in v.items() if a!='fields'} for k,v in results['oracle_residual_contraction'].items()})

if __name__=='__main__':main()
