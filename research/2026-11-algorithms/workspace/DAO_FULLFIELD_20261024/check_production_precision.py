"""Two-sample integration smoke; no optimizer update and no timing claim."""
import json
import numpy as np
import torch
from experiment import System, HERE
from durable_io import atomic_json

system = System('trepan')
data = system.load_data(pilot=True)
model = system.model('large')
saved = torch.load(HERE/'trepan_large/resume.pt', map_location='cpu', weights_only=False)
model.load_state_dict(saved['model'])
model.train()
loss, parts = system.loss(model, data, np.arange(2))
assert loss.dtype == torch.float32 and torch.isfinite(loss)
loss.backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
result = dict(status='passed', samples=2, training_updates=0,
    model_parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
    loss_dtype=str(loss.dtype), differentiable_optics_dtype=str(system.optics32.p0.dtype),
    detached_reference_and_validation_dtype=str(system.optics.p0.dtype), loss=float(loss.detach()),
    parts=parts, timing_claim=False)
atomic_json(HERE/'performance_20261024/production_precision_smoke.json', result)
print(json.dumps(result, indent=2))
