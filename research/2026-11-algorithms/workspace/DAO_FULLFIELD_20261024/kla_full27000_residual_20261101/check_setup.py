"""Small synthetic interface check only; no validation inference or optical replay."""
from pathlib import Path
import sys,json,py_compile
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
from durable_io import atomic_json
for path in ROOT.glob('*.py'):py_compile.compile(str(path),doraise=True)
import models
from network import restore
import numpy as np
import torch
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
old=torch.load(ROOT.parent/'kla_three_improvements_20261101/full_data/candidate.pt',map_location='cpu',weights_only=False)
base,payload=restore(ROOT/'frozen_base.pt')
assert all(torch.equal(v,payload['model'][k[5:]]) for k,v in old['model'].items())
with np.load(ROOT/'selection.npz') as z:
    assert len(z['train'])==27000 and len(np.unique(z['train']))==27000
    assert len(z['indices'])==200 and not np.intersect1d(z['train'],z['indices']).size
head,_=models.restore(ROOT/'frozen_base.pt')
assert sum(p.numel() for p in head.parameters())==522888
assert all(p.dtype==torch.float32 for p in head.parameters())
with torch.no_grad():
    images=base.reference[None].expand(2,-1,-1,-1).clone()
    prior=base.command(images)
    context=dict(prior=prior,expected=images.clone(),available=torch.tensor([True,False],device='cuda'))
    output=head.command(images,context=context)
    assert torch.equal(output,prior)
    head.head[-1].bias.fill_(1.)
    shifted=head.command(images,context=context)
    assert torch.equal(shifted[1],prior[1])
    assert float((shifted-prior).abs().max())<=.0500001
result=dict(passed=True,scope='Synthetic shape/loading/zero-head/fallback contract; no measured cases rescored',
    converted_weights_equal=True,strict_native_model_load=True,correction_parameters=522888,
    network_dtype='float32',zero_head_preserves_prior=True,fallback_preserves_prior=True,
    bounded_correction=True,train=27000,test=200,disjoint=True)
atomic_json(ROOT/'setup_check.json',result)
print(json.dumps(result))
