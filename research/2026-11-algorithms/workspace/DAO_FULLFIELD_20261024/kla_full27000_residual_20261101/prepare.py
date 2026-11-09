"""Prepare the full27000 model-residual combination without recomputing controls."""
from pathlib import Path
import sys,json,shutil
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'kla_three_improvements_20261101'
sys.path.insert(0,str(ROOT.parent))
from durable_io import atomic_json,save_torch

if (ROOT/'protocol.json').exists():raise SystemExit('Already prepared; preserve the registered run')
with np.load(PREVIOUS/'selection.npz') as z:
    train=z['train_full'].copy();rows=z['rows'].copy();ids=z['indices'].copy()
assert len(train)==27000 and len(ids)==200 and not np.intersect1d(train,ids).size
np.savez_compressed(ROOT/'selection.npz',train=train,train_full=train,rows=rows,indices=ids)
a=torch.load(PREVIOUS/'full_data/candidate.pt',map_location='cpu',weights_only=False)
assert a['experiment_arm']=='full_data' and a['epoch']==3
assert all(k.startswith('core.') for k in a['model'])
base=dict(model={k[len('core.'):]:v for k,v in a['model'].items()},reference=a['reference'],scales=a.get('scales'),
          epoch=3,source_checkpoint=str(PREVIOUS/'full_data/candidate.pt'),
          source_version='full27000_oldloss_epoch3',conversion='Remove wrapper core. key prefix only; parameters unchanged')
save_torch(ROOT/'frozen_base.pt',base)
shutil.copy2(PREVIOUS/'full_data/candidate200.npz',ROOT/'baseline200.npz')
shutil.copy2(PREVIOUS/'full_data/candidate200.npz',ROOT/'bounded_control200.npz')
with np.load(ROOT/'baseline200.npz') as z:
    assert np.array_equal(z['indices'],ids)
    np.savez_compressed(ROOT/'control200_grid65.npz',command=z['command'],indices=ids,rows=rows,
                        score=z['score65'],valid=z['valid65'])
    assert int((z['valid']&(z['score']>=.45)).sum())==124
shutil.copy2(PREVIOUS/'bounded_control200.npz',ROOT/'historical_small_data_control200.npz')
shutil.copy2(PREVIOUS/'model_residual/candidate200.npz',ROOT/'historical_old_prior_residual200.npz')
p=json.loads((PREVIOUS/'protocol.json').read_text(encoding='utf-8'))
p.update(task='27000data stronger frozen prior plus27000data model-residual correction',
    arms={'model_residual':'Full27000trained frozen prior + full27000 residual head3epochs'},
    initial_checkpoint=str(ROOT/'frozen_base.pt'),initial_version='full27000_oldloss_epoch3',
    control_checkpoint=str(ROOT/'frozen_base.pt'),control_epoch=3,
    control_passes200=124,initial200_passes=124,
    training_samples=27000,training_per_cohort=9000,epochs_per_arm=3,
    training_selection_source=str(PREVIOUS/'selection.npz')+'::train_full',
    bn='All frozen-prior weights and statistics frozen; residual head has no BatchNorm',
    training_samples_by_arm={'model_residual':27000},updates_by_arm={'model_residual':2532},
    updates_per_arm={'model_residual':2532},total_new_updates=2532,
    reused_control='Saved full27000model epoch3 commands and original refined scores:124/200. No baseline inference or optical replay.',
    budget='Only residual head trains for3epochs/2532updates on27000; frozen full-data base is not retrained',
    training='Fresh zero-output522888parameter head,AdamW3e-5,batch32/micro16,clip5,FP32,oldloss and original optical grid65',
    cache=dict(packets=27200,train=27000,test=200,format='float32 .npy memory maps, per25case durable commit',
        scope='Expected original-camera images at frozen stronger prior; original measured data reused',
        prior_cache_compatible=False,reason='Earlier3200camera predictions used different frozen weights and cannot be mixed',
        expected_bytes=27200*9*128*128*4),
    attribution_limits=['Combination changes the frozen initializer and the residual-head training coverage/update budget.',
        'Primary comparison isolates adding the trained corrector to the SAME full-data prior on the same200cases.',
        'Reused selection validation,one seed; no full3000 or independent generalization claim.'])
p.pop('spatial_head',None)
p['model_residual']['prior']='Frozen full27000_oldloss_epoch3,124/200,unchanged during correction training'
atomic_json(ROOT/'protocol.json',p)
for name in ('camera_context.py','losses.py'):
    shutil.copy2(PREVIOUS/name,ROOT/name)
s=(PREVIOUS/'train_arm.py').read_text(encoding='utf-8')
s=s.replace('Three-epoch old-loss input ablation; automatically score the same fixed200.',
            'Full27000 residual head,3epochs under unchanged old loss,same fixed200.')
s=s.replace("train = plan['train_full'] if ARM == 'full_data' else plan['train']","train = plan['train']")
s=s.replace("source_version='old_wfe_stable_epoch4'","source_version='full27000_oldloss_epoch3'")
s=s.replace("'Full-data has9x training coverage and optimizer updates; not equal compute'", "'Stronger full-data frozen prior;27000sample residual training'")
s=s.replace('KLA three-candidate study;','KLA full27000 residual combination;')
(ROOT/'train_arm.py').write_text(s,encoding='utf-8')
print(json.dumps(dict(prepared=True,train=27000,epochs=3,updates=2532,control_passes=124,cache_packets=27200)))
