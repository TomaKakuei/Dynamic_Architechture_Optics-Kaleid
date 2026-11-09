"""Round3: controlled stopping/merit interventions, identical4round budget."""
from nikon_lab import *
import json
from nikon_closed_loop import NikonFeedback, Session

lab=Lab();start=active_time();adapter=NikonFeedback()
from fast_execution import nikon_execution
adapter.validation_states={int(seed):h.copy() for seed,h in zip(lab.seeds,lab.hidden)}
for seed in adapter.nk.TEST_SEEDS:adapter.validation_states[seed]=adapter.nk.sample(seed)[0]
seeds=list(adapter.nk.TEST_SEEDS)+[int(s) for s in lab.seeds]
adapter.metric_mode='transport'
def policy(session,early=True):
    q=np.zeros(35);raw=session.measure(q,'initial');best=raw['merit']
    pred=np.clip(session.predict(raw),-1.5,1.5)
    trial=session.measure(-pred,'unit_proposal')
    if trial is not None and trial['merit']<best:q,raw,best=-pred,trial,trial['merit']
    reason='four_round_limit'
    for iteration in range(4):
        if early and best<=.04:reason='early_stop';break
        if not session.can_read():reason='reading_budget';break
        direction=cap(adapter.local_direction(raw,session)[None])[0]
        origin=q.copy();accepted=False
        for gain in (1,.5,.25):
            if not session.can_read():break
            trial=session.measure(np.clip(origin+gain*direction,-1.5,1.5),'local_search')
            if trial is not None and trial['merit']<best:
                q,raw,best=trial['command'],trial,trial['merit'];accepted=True;break
        if not accepted:reason='no_accepted_gain';break
    return session.finish(q),reason,best

base_merit=adapter.merit
rows=[]
with nikon_execution():
    for mode in ('baseline','no_early_stop','oracle_merit_no_early_stop'):
        for seed in seeds:
            path=HERE/'round3_cases'/f'{seed}_{mode}.json'
            if path.exists():row=json.loads(path.read_text())
            else:
                adapter.reset(seed)
                # The sole oracle intervention changes the scalar ranking only;
                # update directions still come from the same noisy native packet.
                adapter.merit=(lambda packet,session:max([adapter.audit(packet['command'])['center_abs']]+
                                                         adapter.audit(packet['command'])['edge_diff'])) if mode.startswith('oracle') else base_merit
                session=Session(adapter,seed,budget=20)
                q,reason,merit=policy(session,early=(mode=='baseline'))
                terminal=adapter.audit(q)
                row=dict(seed=seed,mode=mode,cohort='original11' if seed in adapter.nk.TEST_SEEDS else 'validation24',
                         command=q.tolist(),terminal=terminal,counts=session.counts,stop_reason=reason,last_merit=merit,
                         oracle_truth_for_merit=mode.startswith('oracle'))
                atomic(path,row)
            rows.append(row)
        print(mode,{cohort:sum(r['terminal']['fullfield_pass'] for r in rows if r['mode']==mode and r['cohort']==cohort)
                    for cohort in ('original11','validation24')},flush=True)

summary={}
for mode in ('baseline','no_early_stop','oracle_merit_no_early_stop'):
    summary[mode]={}
    for cohort in ('original11','validation24'):
        rr=[r for r in rows if r['mode']==mode and r['cohort']==cohort]
        summary[mode][cohort]=dict(count=len(rr),fullfield_pass=sum(r['terminal']['fullfield_pass'] for r in rr),
            center_pass=sum(r['terminal']['center_pass'] for r in rr),
            mean_worst=float(np.mean([max([r['terminal']['center_abs']]+r['terminal']['edge_diff']) for r in rr])),
            readings=float(np.mean([r['counts']['acquisitions'] for r in rr])),
            false_early_stop_seeds=[r['seed'] for r in rr if r['stop_reason']=='early_stop' and not r['terminal']['fullfield_pass']],
            reasons={reason:sum(r['stop_reason']==reason for r in rr) for reason in sorted({r['stop_reason'] for r in rr})})

with np.load(HERE/'round2_endpoints.npz') as z:j=z['nominal_native_jacobian']
_,s,v=np.linalg.svd(j,full_matrices=False);weak=v[s<.01]
weakq=(lab.q0@weak.T)@weak
projection=dict(keep_strong_only=lab.score(lab.q0-weakq),keep_weak_only=lab.score(weakq),
                note='Oracle pose projections isolate observable and weak subspaces; not inference proposals')
atomic(HERE/'round3.json',dict(round=3,status='completed',network_updates=0,results=summary,
                              pose_projection=projection,effective_seconds=active_time()-start))
print(json.dumps(summary,indent=2));print('projection',{k:brief(v) if isinstance(v,dict) else v for k,v in projection.items()})
