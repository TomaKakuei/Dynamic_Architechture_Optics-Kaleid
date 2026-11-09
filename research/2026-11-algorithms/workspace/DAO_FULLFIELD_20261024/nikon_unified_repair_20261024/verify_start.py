"""Read-only CPU verification of actual optimizer updates and saved baseline."""
from pathlib import Path
import json,numpy as np,torch
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
run=HERE/'training'
saved=torch.load(run/'resume.pt',map_location='cpu',weights_only=False)
steps=sorted({int(v['step']) for v in saved['optimizer']['state'].values()})
assert steps and min(steps)>=10
assert saved['config']['epochs']==4 and saved['config']['parameters']==7121987
assert len(saved['optimizer']['state'])==101
# base and native_optics contain normalization/optical constants, no parameters.
parameters={k:v for k,v in saved['model'].items() if not k.startswith(('base.','native_optics.'))}
assert len(parameters)==101 and sum(v.numel() for v in parameters.values())==7121987
assert all(v.dtype==torch.float32 for v in parameters.values())
with np.load(run/'baseline_validation.npz') as z:
    valid=z['valid'] & np.isfinite(z['fields']).all(1)
    passes=valid & (z['fields'].max(1)<.07)
    moving=(z['initial'].reshape(-1,7,5)!=0).any(2).sum(1)
    baseline=dict(count=len(passes),passes=int(passes.sum()),rate=float(passes.mean()),
        all7_count=int((moving==7).sum()),all7_passes=int(passes[moving==7].sum()),
        invalid=int((~valid).sum()))
    assert baseline['count']==6000 and baseline['all7_count']==1333
report=json.loads((run/'baseline.json').read_text())
assert baseline['passes']==report['full_field_passes']
result=dict(updated=datetime.now(timezone.utc).isoformat(),actual_saved_optimizer_steps=steps,
    trainable_parameter_tensors_with_optimizer_state=101,parameters=7121987,network_dtype='float32',
    baseline=baseline,baseline_is_after_training_only_readout_fit=True,
    trained_four_epoch_result='pending',estimated_remaining_minutes=[25,35])
(HERE/'startup_verified.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))

if __name__=='__main__':pass
