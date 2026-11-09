"""Frozen Nikon full-field model with observation-only feedback, no training.

Center uses absolute reference-sphere PTTD; edges use difference from the
fixed corrected bias. Exact state is confined to simulation and final scoring.
The historical20reading/4feedback-round/.04stop/.25local-step settings remain.
"""
import argparse
import json
import sys
import numpy as np
import torch
from experiment import System, HERE, ROOT, atomic, active_time

sys.path.insert(0,str(ROOT/'oneshot_resources_11_20261014'))
from study_common import Session, new_policy

OUT=HERE/'nikon_debug_20261024'

class NikonFeedback:
    name='nikon'; dofs=35; forward_count=1
    def __init__(self,variant='large'):
        self.system=System('nikon'); self.nk=self.system.legacy.nikon35
        self.model=self.system.model(variant).eval()
        if variant=='large':
            self.model.load_state_dict(torch.load(HERE/'nikon_large/candidate.pt',map_location='cpu',weights_only=False)['model'])
        self.model.requires_grad_(False)
        self.native=self.system.legacy.NikonTensorOptics(nodes=self.nk.NODES,
            fields=self.nk.base.corrected.CONTROL_FIELDS_MM,device='cuda')
        self.metric_mode='diff'; self.calibration=self.make_calibration()
        self.prepare_field_transport()

    def residual(self,q):
        audit=self.system.optics.metrics(q)
        native=self.native.metrics(q)
        # Each field has unit RMS weighting despite different sample counts.
        return torch.cat((audit['absolute'][:,0]/np.sqrt(audit['absolute'].shape[-1]),
                          native['difference'][:,1:].flatten(1)/np.sqrt(len(self.nk.NODES))),dim=1)

    def make_calibration(self):
        h=1e-4
        q=torch.eye(35,device='cuda',dtype=torch.float64)*h
        with torch.no_grad():
            j=((self.residual(q)-self.residual(-q))/(2*h)).T.cpu().numpy()
        u,s,vt=np.linalg.svd(j,full_matrices=False)
        self.singular_values=s
        return (vt.T*(s/(s*s+1e-4)))@u.T

    def prepare_field_transport(self):
        # Native sensor fields are +/-0.05mm; reported edges are +/-0.1mm.
        # Use only fixed nominal optical response to transport measured
        # residuals to reporting fields. No hidden pose enters this mapping.
        def observed(q):
            a=self.system.optics.metrics(q);n=self.native.metrics(q)
            return torch.cat((a['absolute'][:,0]/np.sqrt(a['absolute'].shape[-1]),
                              n['difference'].flatten(1)/np.sqrt(49)),dim=1)
        def reported(q):
            m=self.system.optics.metrics(q)
            return torch.cat((m['absolute'][:,:1],m['difference'][:,1:]),dim=1).flatten(1)
        zero=torch.zeros((1,35),dtype=torch.float64,device='cuda');h=1e-4
        dq=torch.eye(35,dtype=torch.float64,device='cuda')*h
        with torch.no_grad():
            self.observed_reference=observed(zero)[0].cpu().numpy()
            self.reported_reference=reported(zero)[0].cpu().numpy()
            jo=((observed(dq)-observed(-dq))/(2*h)).T.cpu().numpy()
            jr=((reported(dq)-reported(-dq))/(2*h)).T.cpu().numpy()
        u,s,vt=np.linalg.svd(jo,full_matrices=False)
        self.nominal_inverse=(vt.T*(s/(s*s+1e-4)))@u.T
        self.reported_response=jr

    def inferred_delta(self,raw):
        observed=np.r_[raw['center']/np.sqrt(len(raw['center'])),raw['network'].ravel()/np.sqrt(49)]
        return self.nominal_inverse@(observed-self.observed_reference)

    def reset(self,seed):
        self.seed=seed
        self.hidden=(self.validation_states[seed].copy() if hasattr(self,'validation_states') else self.nk.sample(seed)[0])
        self.plant=self.nk.Plant(self.hidden,noise_seed=seed,noise_std=.015,score_center=True)

    def acquire(self,command,index):
        return self.plant.measure(command),dict(field_traces=4,ray_launches=348,noise_seed=self.seed)

    def merit(self,packet,session):
        raw=packet['raw']; center=float(np.sqrt(np.mean(raw['center']**2)))
        if self.metric_mode=='center':return center
        if self.metric_mode=='transport':
            predicted=(self.reported_reference+self.reported_response@self.inferred_delta(raw)).reshape(3,-1)
            return max(center,float(np.sqrt(np.mean(predicted**2,axis=1)).max()))
        return max(center,float(np.sqrt(np.mean(raw['network'][1:]**2,axis=1)).max()))

    def predict(self,raw):
        raw=np.asarray(raw['network']).reshape(1,3,49)
        image=self.nk.grid(raw)
        with torch.inference_mode():
            pred=self.model(torch.as_tensor(image,device='cuda'),torch.as_tensor(raw.astype(np.float32),device='cuda'))
            base=self.model.base if hasattr(self.model,'system') else self.model
            physical=pred*base.target_std+base.target_mean
        return physical[0].cpu().numpy().astype(float)

    def allowed(self,start,end):
        return bool(np.isfinite(end).all() and abs(end).max()<=1.5000001)

    def local_direction(self,packet,session):
        raw=packet['raw']
        if self.metric_mode=='transport':return -self.inferred_delta(raw)
        r=np.r_[raw['center']/np.sqrt(len(raw['center'])),raw['network'][1:].ravel()/np.sqrt(49)]
        return -self.calibration@r

    def accepted(self,old,new,dq):pass

    def audit(self,command):
        q=torch.as_tensor((self.hidden+command)[None],dtype=torch.float64,device='cuda')
        with torch.no_grad():
            w,valid,_=self.system.metrics(q,q)
        w=w[0].cpu().numpy(); good=bool(valid[0])
        return dict(center_abs=float(w[0]),edge_diff=w[1:].tolist(),field_mean=float(w.mean()),
                    fullfield_pass=bool(good and w.max()<.07),center_pass=bool(good and w[0]<.07),valid=good)

def run(adapter,seed,method):
    adapter.reset(seed);adapter.metric_mode=('center' if method=='original_center_loop' else
                                            'transport' if method=='diff_calibrated' else 'diff')
    policy={'one_shot':'one_shot','original_center_loop':'looping','diff_loop':'looping','diff_local':'local_search',
            'diff_calibrated':'local_search'}[method]
    session=Session(adapter,seed,budget=20)
    command=new_policy(session,policy)
    terminal=adapter.audit(command)
    return dict(seed=seed,method=method,terminal=terminal,command=command.tolist(),
                counts=session.counts,events=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in e.items()} for e in session.events],
                observation_only_decisions=True,truth_used_for_control=False)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--validation24',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);start=active_time();adapter=NikonFeedback()
    from fast_execution import nikon_execution
    seeds=list(adapter.nk.TEST_SEEDS);seeds=seeds[:1] if args.pilot else seeds
    methods=['one_shot','original_center_loop','diff_loop','diff_local','diff_calibrated'];rows=[]
    case_dir=OUT/'closed_loop_cases'
    summary_name='closed_loop_pilot.json' if args.pilot else 'closed_loop_summary.json'
    if args.validation24:
        with np.load(OUT/'small_sample_endpoints.npz') as z:
            ix=np.r_[96:104,128:136,160:168]
            seeds=[int(v) for v in z['ids'][ix]]
            adapter.validation_states={int(z['ids'][i]):z['q'][i].copy() for i in ix}
        methods=['one_shot','diff_calibrated'];case_dir=OUT/'validation24_cases';summary_name='validation24_summary.json'
    # Check forward sensor units/reference on one actual state, without noise.
    q,_=adapter.nk.sample(seeds[0]); plant=adapter.nk.Plant(q,noise_std=0.,score_center=True)
    with nikon_execution():
        measured=plant.measure(np.zeros(35))
        with torch.no_grad():
            t=torch.tensor(q[None],dtype=torch.float64,device='cuda')
            nd=adapter.native.metrics(t)['difference'][0].cpu().numpy()
            ca=adapter.system.optics.metrics(t)['absolute'][0,0].cpu().numpy()
        agreement=dict(native_diff_max_error=float(np.max(np.abs(nd-measured['network']))),
                       center_abs_max_error=float(np.max(np.abs(ca-measured['center']))))
        assert agreement['native_diff_max_error']<1e-5 and agreement['center_abs_max_error']<1e-5,agreement
        atomic(OUT/'sensor_agreement.json',agreement)
        for seed in seeds:
            for method in methods:
                path=case_dir/f'{seed}_{method}.json'
                if path.exists():row=json.loads(path.read_text())
                else:
                    row=run(adapter,seed,method);atomic(path,row)
                rows.append(row)
                print(seed,method,row['terminal'],'reads',row['counts']['acquisitions'],flush=True)
                atomic(OUT/'closed_loop_state.json',dict(status='running',completed=len(rows),planned=len(seeds)*len(methods),seed=seed,method=method))
    summary={}
    for method in methods:
        rr=[r for r in rows if r['method']==method]
        summary[method]=dict(count=len(rr),fullfield_pass=sum(r['terminal']['fullfield_pass'] for r in rr),
             center_pass=sum(r['terminal']['center_pass'] for r in rr),
             field_mean=float(np.mean([r['terminal']['field_mean'] for r in rr])),
             per_field_mean=np.mean([[r['terminal']['center_abs']]+r['terminal']['edge_diff'] for r in rr],axis=0).tolist(),
             acquisitions_mean=float(np.mean([r['counts']['acquisitions'] for r in rr])),
             invalid=sum(not r['terminal']['valid'] for r in rr))
    atomic(OUT/summary_name,
        dict(no_training=True,seeds=seeds,metric='center ABS + edges DIFF <0.07waves',sensor_agreement=agreement,
             methods=summary,calibration_singular_values=adapter.singular_values.tolist(),effective_seconds=active_time()-start,
             observed_fields_mm=list(adapter.native.fields),evaluated_fields_mm=list(adapter.system.optics.fields),
             independent_test=False,scope=('First8 original validation rows per3groups,24total; not a fresh test set'
                                           if args.validation24 else 'Original11 diagnostic cases; protocol comparison, not fresh generalization')))
    atomic(OUT/'closed_loop_state.json',dict(status='completed',completed=len(rows),planned=len(rows)))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
