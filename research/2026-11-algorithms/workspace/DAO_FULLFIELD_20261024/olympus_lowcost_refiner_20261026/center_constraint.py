from bridge import *


@torch.no_grad()
def main():
    if (RUN/'center_gram.npy').exists():return
    obj=Objective();h=.001;columns=[]
    for start in range(0,35,5):
        indices=np.arange(start,min(start+5,35));q=np.eye(35)[indices]*h
        a,ma,_=obj.center.phase(torch.as_tensor(q,device='cuda'))
        b,mb,_=obj.center.phase(torch.as_tensor(-q,device='cuda'))
        mask=ma&mb;w=mask.double();d=(a-b)/(2*h);A=obj.center.low_design
        gram=torch.einsum('nr,bfn,ns->bfrs',A,w,A);rhs=torch.einsum('nr,bfn,bfn->bfr',A,w,d)
        coef=torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
        v=torch.where(mask,d-torch.einsum('nr,bfr->bfn',A,coef),0)/w.sum(-1,keepdim=True).sqrt()
        columns.append(v.flatten(1).cpu().numpy())
    J=np.concatenate(columns).T;np.save(RUN/'center_gram.npy',J.T@J)
    common.atomic(RUN/'center_constraint.json',dict(definition='Nominal known-model linearized center PTTD phase change induced by departure from neural prior',
        radius_waves=.01,not_a_guarantee_for_unknown_true_endpoint=True,training_or_validation_labels_used=False))


if __name__=='__main__':main()
