"""Short original11 closed-loop comparison after actual4epoch training."""
from short_train import TrialSystem,HERE,exp,torch,np
from nikon_closed_loop import NikonFeedback,run
import argparse,json
p=argparse.ArgumentParser();p.add_argument('--idea',required=True);a=p.parse_args()
folder=HERE/a.idea
s=TrialSystem(a.idea);adapter=NikonFeedback();adapter.model=s.model('large')
saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
adapter.model.load_state_dict(saved['model']);adapter.model.eval().requires_grad_(False)
from fast_execution import nikon_execution
rows=[]
with nikon_execution():
    for seed in adapter.nk.TEST_SEEDS:
        for method in ('one_shot','diff_calibrated'):
            path=folder/'loop_cases'/f'{seed}_{method}.json'
            if path.exists():row=json.loads(path.read_text())
            else:row=run(adapter,seed,method);exp.atomic(path,row)
            rows.append(row)
            print(seed,method,row['terminal'],flush=True)
summary={}
for method in ('one_shot','diff_calibrated'):
    r=[x for x in rows if x['method']==method]
    summary[method]=dict(count=len(r),fullfield_pass=sum(x['terminal']['fullfield_pass'] for x in r),
        center_pass=sum(x['terminal']['center_pass'] for x in r),
        field_mean=float(np.mean([x['terminal']['field_mean'] for x in r])),
        acquisitions=float(np.mean([x['counts']['acquisitions'] for x in r])))
exp.atomic(folder/'loop_summary.json',dict(epoch=saved['epoch'],source=str(folder/'last.pt'),results=summary,
    scope='Original11 only; center ABS and edge DIFF; noisy observations only in control; no independent-test claim'))
print(json.dumps(summary,indent=2))
