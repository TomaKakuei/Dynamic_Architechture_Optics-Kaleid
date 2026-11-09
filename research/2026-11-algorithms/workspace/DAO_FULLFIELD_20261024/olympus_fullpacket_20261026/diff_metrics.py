"""Center ABS and edge signed-wavefront DIFF from the nominal prescription.

Scoring only: hidden initial poses do not enter the camera inverse.
"""
from support import *
import json


class DiffMetrics:
    def __init__(self):
        self.obj=Objective()
        with torch.no_grad():
            self.nominal,self.mask,self.chief=self.obj.full.phase(torch.zeros((1,35),device='cuda',dtype=torch.float64))
            self.nominal_abs=self.obj.full.high_order(torch.zeros((1,35),device='cuda',dtype=torch.float64))[0].clamp_min(0).sqrt()[0].cpu().numpy()

    @torch.no_grad()
    def score(self,initial,command):
        q=torch.as_tensor(np.asarray(initial)+np.asarray(command),device='cuda').double()
        start=torch.as_tensor(initial,device='cuda').double()
        if q.ndim==1:q=q[None];start=start[None]
        phase,valid,chief=self.obj.full.phase(q)
        mask=valid&self.mask
        difference=phase-self.nominal
        difference[:,0]=phase[:,0]
        design=self.obj.full.low_design;w=mask.double()
        gram=torch.einsum('nr,bfn,ns->bfrs',design,w,design)
        rhs=torch.einsum('nr,bfn,bfn->bfr',design,difference,w)
        coef=torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
        residual=torch.where(mask,difference-torch.einsum('nr,bfr->bfn',design,coef),0)
        rms=(residual.square().sum(-1)/w.sum(-1)).sqrt().cpu().numpy()
        good=(self.obj.path_valid(q,start)&chief.all(1)&(w.mean(-1)>=.2).all(1)&torch.isfinite(residual).all(-1).all(-1)).cpu().numpy()
        return rms,good


def main():
    out=HERE/'pilot'/'diff_metrics.json'
    if out.exists():print(out.read_text());return
    summary=json.loads((HERE/'pilot/summary.json').read_text())
    with np.load(SOURCE/'training/selected_validation.npz') as z:old={k:z[k].copy() for k in z.files}
    rows=summary['rows'];slots=[int(np.where(old['index']==r['case'])[0][0]) for r in rows]
    metric=DiffMetrics();before,bvalid=metric.score(old['initial'][slots],old['command'][slots])
    after,avalid=metric.score(old['initial'][slots],np.array([r['command'] for r in rows]))
    result=dict(definition='Center ABS; edge phase(q)-phase(0) on common pupil, then PTTD removal; .07 wave threshold at all nine fields; original mechanical validity',
                nominal_abs=metric.nominal_abs.tolist(),rows=[])
    for k,r in enumerate(rows):
        result['rows'].append(dict(case=r['case'],before=before[k].tolist(),after8=after[k].tolist(),
            before_valid=bool(bvalid[k]),after_valid=bool(avalid[k]),
            before_pass=bool(bvalid[k] and np.all(before[k]<.07)),after_pass=bool(avalid[k] and np.all(after[k]<.07))))
    for name,values,good in [('baseline',before,bvalid),('after8',after,avalid)]:
        result[name]=dict(passes=int((good&(values<.07).all(1)).sum()),invalid=int((~good).sum()),
             center_mean=float(values[:,0].mean()),edge_diff_mean=float(values[:,1:].mean()),max_field_mean=float(values.max(1).mean()))
    common.atomic(out,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
