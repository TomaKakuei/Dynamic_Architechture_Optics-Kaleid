"""Train-only error basis and fixed diagnostic/confirmation selection."""
from bridge import *


@torch.no_grad()
def main():
    old=baseline()
    if not (RUN/'selection.json').exists():
        ids=old['index'];valid=old['valid'];v=old['mixed_wrms']
        rng=np.random.default_rng(2026102611)
        masks=dict(pass6=valid&(v<.07).all(1),center_fail2=valid&(v[:,0]>=.07),
                   edge_fail2=valid&(v[:,0]<.07)&(v[:,1:]>=.07).any(1),invalid2=~valid)
        chosen=[];groups={}
        for name,mask in masks.items():
            n=6 if name=='pass6' else 2
            pool=ids[mask&~np.isin(ids,[7111,7007,7049,7363])]
            take=rng.choice(pool,n,replace=False).tolist();groups[name]=take;chosen+=take
        pool=ids[~np.isin(ids,chosen+[7111,7007,7049,7363])]
        confirm=np.random.default_rng(2026102612).choice(pool,32,replace=False).tolist()
        common.atomic(RUN/'selection.json',dict(development=chosen,groups=groups,confirmation=confirm,
            development_scope='6 existing passes,2 center failures,2 edge-only failures,2 invalid; not prevalence representative',
            confirmation_scope='Random32 from remaining original500, frozen before refiner development; reused validation, not a fresh independent dataset'))
    if (RUN/'calibration.npz').exists():return
    state('calibrating_train_error_basis',training_cases=320)
    model,_=restore(REPAIR/'training/last.pt');blocks=Blocks();errors=[];trainids=[]
    for b in np.linspace(0,279,32,dtype=int):
        x,q=blocks.read('joint',int(b));pred=-model.command(torch.as_tensor(x[:10].astype(np.float32),device='cuda')).cpu().numpy()
        errors.append(q[:10]-pred);trainids.extend((np.arange(10)+int(b)*25).tolist())
    errors=np.concatenate(errors);cov=errors.T@errors/len(errors)
    cov=.95*cov+.05*np.trace(cov)/35*np.eye(35)
    # Rank directions by expected endpoint optical error under the train-only
    # covariance. The optical sensitivity uses the known nominal prescription.
    obj=Objective();h=.001;zero=torch.zeros((1,35),device='cuda',dtype=torch.float64)
    f0,m0,_=obj.full.phase(zero);design=obj.full.low_design
    columns=[]
    for start in range(0,35,5):
        indices=np.arange(start,min(start+5,35));q=np.eye(35)[indices]*h
        plus,mp,_=obj.full.phase(torch.as_tensor(q,device='cuda'))
        minus,mm,_=obj.full.phase(torch.as_tensor(-q,device='cuda'))
        mask=mp&mm&m0;w=mask.double();der=(plus-minus)/(2*h)
        gram=torch.einsum('nr,bfn,ns->bfrs',design,w,design)
        rhs=torch.einsum('nr,bfn,bfn->bfr',design,w,der)
        co=torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
        response=torch.where(mask,der-torch.einsum('nr,bfr->bfn',design,co),0)
        columns.append((response/(9*w.sum(-1,keepdim=True)).sqrt()).flatten(1).cpu().numpy())
    J=np.concatenate(columns).T;G=J.T@J
    ev,U=np.linalg.eigh(cov);sqrtC=(U*np.sqrt(np.maximum(ev,1e-12)))@U.T
    importance,V=np.linalg.eigh(sqrtC@G@sqrtC)
    candidates=sqrtC@V[:,::-1];basis,_=np.linalg.qr(candidates)
    np.savez_compressed(RUN/'calibration.npz',basis=basis,covariance=cov,optical_gram=G,
                        importance=importance[::-1],training_ids=np.array(trainids),errors=errors)
    common.atomic(RUN/'calibration.json',dict(training_cases=len(errors),training_ids=trainids,
        error_rms=float(np.sqrt(np.mean(errors**2))),basis_definition='QR of training covariance square-root times dominant nominal optical-risk eigenvectors',
        first4_optical_risk_fraction=float(importance[-4:].sum()/importance.sum()),
        first8_optical_risk_fraction=float(importance[-8:].sum()/importance.sum()),
        validation_labels_used_in_basis=False,network_training_updates=0))
    state('prepared',training_cases=320,development_cases=12,confirmation_cases=32)


if __name__=='__main__':main()
