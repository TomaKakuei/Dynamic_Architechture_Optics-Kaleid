"""Register three prospective candidates and reuse the same200/control."""
from pathlib import Path
import json,sys,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'kla_input_ablation3ep_200_20261031'
sys.path.insert(0,str(ROOT.parent))
from durable_io import atomic_json
from models import ARMS,SOURCE

if (ROOT/'protocol.json').exists():raise SystemExit('Already prepared; preserve existing protocol.')
for name in ('baseline200.npz','control200_grid65.npz','bounded_control200.npz'):
    shutil.copy2(PREVIOUS/name,ROOT/name)
with np.load(PREVIOUS/'selection.npz') as z:selection={k:z[k].copy() for k in z.files}
with np.load(SOURCE/'cache/metadata.npz') as z:
    full=np.where(z['indices']<9000)[0]
    assert len(full)==27000
    assert all((z['cohort'][full]==c).sum()==9000 for c in range(3))
assert not np.intersect1d(full,selection['indices']).size
selection['train_full']=full
np.savez_compressed(ROOT/'selection.npz',**selection)
p=json.loads((PREVIOUS/'protocol.json').read_text(encoding='utf-8'))
shutil.copy2(p['initial_checkpoint'],ROOT/'frozen_base.pt')
for key in ('masking','field_indices','adapter','raw_adapter_input','difference_adapter_input','information_scope'):
    p.pop(key,None)
p.update(task='Three prospective KLA improvements with unchanged old loss',arms=ARMS,
    initial_checkpoint=str(ROOT/'frozen_base.pt'),source_data=str(SOURCE),
    total_new_updates=282+282+2532,
    training_samples_by_arm=dict(spatial_head=3000,model_residual=3000,full_data=27000),
    training_samples=dict(spatial_head=3000,model_residual=3000,full_data=27000),
    training_per_cohort=dict(spatial_head=1000,model_residual=1000,full_data=9000),
    updates_by_arm=dict(spatial_head=282,model_residual=282,full_data=2532),
    updates_per_arm=dict(spatial_head=282,model_residual=282,full_data=2532),
    reused_control='Archived3000sample3epoch original9field control120/200. Full_data has9x more updates; no equal-budget claim for that comparison.',
    budget='3complete epochs per arm; full_data has9x examples and about9x updates, disclosed rather than claimed equal-compute',
    spatial_head='Read pretrained256channel4x4 CNN spatial features before global averaging through4096->128->40 zero-initialized logit head; original weights trainable',
    model_residual=dict(prior='Frozen original selected model, unchanged across3epochs',
        context='Observed noisy9field image minus original native-camera expected image at inferred initial pose -prior; no true pose used',
        correction='New convolutional residual head conditioned on camera residual and40Dprior; output cap0.05*tanh(raw/0.05); known command bounds[-1,1]',
        training='Only correction head trained, original bounded-WRMS+0.05pose+10mechanical; no image-fit loss',
        camera='Original2048pupil/4096FFT, fixed diffuser/exposure/PRNU/DSNU; original4096/8192 bandwidth fallback',
        failure_policy='Model-domain failures keep the frozen prior, report failures and retain all cases; infrastructure errors halt',
        deployment='One original acquisition, frozen neural prior + one model render + correction head; extra computation, not a single neural pass'),
    data_generation=False,new_acquisitions=0,
    training='Same seed/orders,lr3e-5,AdamW,clip5,FP32 network,FP64optics,traininggrid65; BNfrozen for original core',
    inference='One observed packet per case; model_residual adds model computation and a correction head, no observed feedback',
    attribution_limits=[
        'These are three practical candidates, not a one-factor equal-compute ablation.',
        'Full-data changes coverage and optimizer-update count jointly.',
        'Spatial readout changes architecture and capacity.',
        'Model residual uses a frozen prior, extra model computation and newly trained correction head.',
        'Same200reused selection cases, one seed, not independent confirmation.'])
atomic_json(ROOT/'protocol.json',p)

s=(PREVIOUS/'train_arm.py').read_text(encoding='utf-8')
s=s.replace('from input_network import restore','from models import restore, cache_context')
s=s.replace("train, rows = plan['train'], plan['rows']", "train = plan['train_full'] if ARM == 'full_data' else plan['train']\n    rows = plan['rows']")
s=s.replace('input_ablation_arm=ARM','experiment_arm=ARM')
s=s.replace('cmd = model.command(observed)',"cmd = model.command(observed, context=cache_context(ix) if ARM == 'model_residual' else None)")
s=s.replace("device='cuda')).cpu().numpy() for first in range(0, len(rows), 16)])",
    "device='cuda'), context=cache_context(plan['indices'][first:first+16]) if ARM == 'model_residual' else None).cpu().numpy() for first in range(0, len(rows), 16)])")
s=s.replace("inference='one packet, one prediction, no loop'", "inference=protocol['inference']")
s=s.replace("'One seed and three epochs; masked inputs warm-start the full-field model'", "'Full-data has9x training coverage and optimizer updates; not equal compute'")
s=s.replace("'Same-packet feature changes cannot establish physical measurement sufficiency'", "'Model residual adds computation from one observation, not a second physical observation'")
s=s.replace('KLA input three-epoch ablation;', 'KLA three-candidate study;')
s=s.replace("    del model\n    torch.cuda.empty_cache()", "    if ARM == 'model_residual':\n        cache_summary=json.loads((OUT/'cache_complete.json').read_text())\n        atomic_json(OUT/'render_fallbacks.json',cache_summary['fallbacks'])\n    del model\n    torch.cuda.empty_cache()")
(ROOT/'train_arm.py').write_text(s,encoding='utf-8')

s=(PREVIOUS/'run_suite.py').read_text(encoding='utf-8')
s=s.replace('four-arm','three-candidate').replace('total_arms=4','total_arms=3')
s=s.replace('KLA input ablation, old loss fixed: each new arm3epochs, same3000train and200validation.',
    'KLA three candidates:3epochs each,3000/3000/27000train,same200validation,oldloss fixed.')
s=s.replace('one packet, one prediction; no loop','one packet; model-residual adds model rendering and learned correction, no observed feedback')
s=s.replace('one archived packet, one prediction.','one archived packet; model-residual uses extra model computation.')
s=s.replace("('center_only', 'three_fields'), ('three_fields', 'nine_control'),\n            ('nine_control', 'nine_raw_adapter'), ('nine_raw_adapter', 'nine_diff_adapter')",
    "('nine_control', 'spatial_head'), ('nine_control', 'model_residual'), ('nine_control', 'full_data')")
needle="        with (out/'stdout.log').open('a', encoding='utf-8') as stdout"
insert="""        if arm == 'model_residual' and not (out/'cache_complete.json').exists():
            with (out/'cache.stdout.log').open('a',encoding='utf-8') as co, (out/'cache.stderr.log').open('a',encoding='utf-8') as ce:
                prep=subprocess.Popen([sys.executable,'-u',str(ROOT/'prepare_camera_cache.py')],
                    cwd=str(ROOT),stdout=co,stderr=ce,creationflags=subprocess.CREATE_NO_WINDOW)
                state('preparing_model_residual',arm=arm,child_pid=prep.pid,completed_arms=list(results),total_arms=3)
                rc=prep.wait()
            if rc != 0 or not (out/'cache_complete.json').exists():
                state('cache_failed_or_stopped',arm=arm,child_exit_code=rc,completed_arms=list(results));return
"""
assert needle in s
s=s.replace(needle,insert+needle)
(ROOT/'run_suite.py').write_text(s,encoding='utf-8')
print(json.dumps(dict(prepared=True,arms=list(ARMS),train_counts=p['training_samples_by_arm'],updates=p['updates_by_arm'])))
