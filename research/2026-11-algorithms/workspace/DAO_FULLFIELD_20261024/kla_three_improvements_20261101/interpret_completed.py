"""Saved-result interpretation only; no training, inference or optical replay."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from durable_io import atomic_json
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def arrays(path):
    with np.load(path) as z:return {k:z[k].copy() for k in z.files}

r=read(ROOT/'result.json')
assert r['complete'] and r['completed_arms']==3
training={a:read(ROOT/a/'training_complete.json') for a in r['arms']}
detail={a:read(ROOT/a/'result.json') for a in r['arms']}
for arm,expected in [('spatial_head',282),('model_residual',282),('full_data',2532)]:
    assert training[arm]['epochs']==3 and training[arm]['updates']==expected
base=arrays(ROOT/'baseline200.npz')
control=arrays(ROOT/'bounded_control200.npz')
all_arrays={'initial':base,'bounded_control':control}
all_arrays.update({a:arrays(ROOT/a/'candidate200.npz') for a in r['arms']})
cohorts={}
for name,a in all_arrays.items():
    cohorts[name]={}
    for c in range(3):
        chosen=(a['rows']//1000)==c
        good=a['valid']&(a['score']>=.45)
        cohorts[name][str(c)]=dict(count=int(chosen.sum()),passes=int(good[chosen].sum()),
            q_mean=float(a['score'][chosen].mean()),wrms_mean=float(a['wrms'][chosen].mean()))
cache=read(ROOT/'model_residual/cache_complete.json')
active_train=sum(t['active_train_seconds'] for t in training.values())
active_eval=sum(x['evaluation_active_seconds'] for x in r['arms'].values())
wrms_change=100*(1-r['arms']['full_data']['candidate']['wrms_mean']/r['bounded_control']['wrms_mean'])
delta=all_arrays['model_residual']['command']-base['command']
now=datetime.now(timezone.utc).isoformat()
review=dict(status='complete_and_interpreted',updated=now,completed_arms=3,
    evidence='Read existing result/history/prediction arrays; no independent replay',
    control=r['bounded_control'],initial=r['baseline'],
    arms={a:x['candidate']|dict(gained_vs_control=x['direct_vs_control']['gained'],
          lost_vs_control=x['direct_vs_control']['lost'],training_samples=detail[a]['actual_training_samples'],
          updates=training[a]['updates']) for a,x in r['arms'].items()},
    cohorts=cohorts,full_data_mean_wrms_reduction_vs_control_percent=wrms_change,
    residual_correction=dict(mean_abs=float(abs(delta).mean()),max_abs=float(abs(delta).max()),
        cases_any_coordinate_near_cap=int((abs(delta).max(1)>=.0499).sum()),
        training_render_fallbacks=cache['training_fallbacks'],validation_render_fallbacks=cache['validation_fallbacks']),
    cost=dict(training_seconds=active_train,evaluation_seconds=active_eval,
        cached_model_render_seconds=cache['render_seconds'],
        total_recorded_compute_seconds=active_train+active_eval+cache['render_seconds'],
        scope='Excludes interruption downtime, initialization, cache IO and initial prior inference; cached residual evaluation is not uncached deployment latency'),
    conclusion=[
        'Full27000training is best in this trial:124/200=62%,versus matched-old3000control120/200=60%;10gains6losses,net4.',
        'Mean Q rises0.434375->0.453996 and mean WRMS falls0.101998->0.094483waves; quality change is broader than the4threshold crossings.',
        'This supports the full-data continuation recipe as a promising candidate, but does not isolate sample coverage from9xmore optimizer updates.',
        'Spatial readout115/200 loses5net versus control; no advantage established.',
        'Model-residual119/200 improves the frozen initial111/200 but remains1below the ordinary fine-tuning control and adds native-camera inference computation.',
        'All routes have0invalid endpoints; all3200model context renders succeeded without fallback.',
        'Small gain on reused200selection cases, one seed; not a confirmed full3000 improvement or independent generalization result.',
        'Keep existing full3000 incumbent and preserve this candidate separately; no automatic deployment or full3000 evaluation.'],
    one_observed_packet=True,loss_changed=False,remaining_seconds=0,
    queue_and_final_worker_exited=True,automatic_followup=False)
atomic_json(ROOT/'assistant_review.json',review)
lines=[
    'KLA THREE CANDIDATES COMPLETE and interpreted2026-11-01.',
    'Finished2026-11-01 12:03 PST. Old loss unchanged; same200reused selection cases.',
    'Original control120/200=60%,initial checkpoint111/200=55.5%.',
    'All3epochs. Spatial/residual3000samples282updates;full-data27000samples2532updates.',
    'Original Q2000>=.45,path/support,129/257refinement retained; allinvalid0.', '',
    'Candidate | passes/200 | percent | mean WRMS waves | gains/losses vs control']
for a,x in r['arms'].items():
    m,p=x['candidate'],x['direct_vs_control']
    lines.append(f"{a} | {m['passes']}/200 | {100*m['rate']:.1f} | {m['wrms_mean']:.8f} | +{p['gained']}/-{p['lost']}")
lines+=['','Interpretation:']+review['conclusion']
lines+=['',f'Full-data mean WRMS reduction versus control:{wrms_change:.3f}%.',
    f'Recorded training{active_train:.2f}s,evaluation{active_eval:.2f}s,model rendering{cache["render_seconds"]:.2f}s.',
    'Interrupted after154residual updates; resumed exact checkpoint and reused all3200contexts; no numerical changes.',
    'Interrupted downtime excluded. Models/sources/failure evidence remain preserved.',
    'One observed packet throughout. Residual route is frozenNN+nativecamera+correctionNN,not one pure neural pass.',
    'No backend running,EST=0. No full3000 or further training queued.']
(ROOT/'FINAL_REVIEW.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
for path in [ROOT/'state.json',ROOT/'result.json']:
    x=read(path);x.update(assistant_interpretation='completed',assistant_review_file='assistant_review.json',assistant_review_updated=now)
    atomic_json(path,x)
for arm in r['arms']:
    path=ROOT/arm/'state.json';x=read(path)
    x.update(assistant_interpretation='completed_in_suite',assistant_review_file='../assistant_review.json')
    atomic_json(path,x)
note=('LATEST THREE KLA CANDIDATES COMPLETE2026-11-01: oldloss/same200,control120; '
      'spatial115,model-residual119,full27000data124(62%). Full-data10gains6losses,WRMS-7.37%,invalid0. '
      'Best this pilot but coverage and9xupdates confounded; no full3000 confirmation/adoption. '
      'Residual3200renders all succeeded,requires extra model computation from one observation. '
      'No background,EST=0. Read kla_three_improvements_20261101/FINAL_REVIEW.txt and assistant_review.json.\n\n')
for path in (ROOT/'HANDOFF.txt',ROOT.parent/'HANDOFF.txt'):
    old=path.read_text(encoding='utf-8')
    if not old.startswith('LATEST THREE KLA CANDIDATES COMPLETE2026-11-01:'):
        path.write_text(note+'--- Historical context retained ---\n'+old,encoding='utf-8')
print(json.dumps(dict(completed=True,wrms_reduction_percent=wrms_change,cohorts=cohorts,
    cost=review['cost'],residual_correction=review['residual_correction'])))
