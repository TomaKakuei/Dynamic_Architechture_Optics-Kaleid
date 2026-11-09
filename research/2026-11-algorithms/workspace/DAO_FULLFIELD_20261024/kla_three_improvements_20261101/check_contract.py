"""Synthetic-input startup checks, without replaying training/validation cases."""
from pathlib import Path
import json,torch
ROOT=Path(__file__).resolve().parent
from models import restore,ARMS
torch.set_num_threads(1)
p=json.loads((ROOT/'protocol.json').read_text())
result={}
for arm in ARMS:
    torch.manual_seed(p['model_seed'])
    model,payload=restore(p['initial_checkpoint'],arm=arm,device='cpu')
    info=dict(trainable_parameters=sum(t.numel() for t in model.parameters() if t.requires_grad))
    with torch.no_grad():
        rng=torch.Generator().manual_seed(1933619)
        x=torch.as_tensor(payload['reference'])[None]+80*torch.randn((2,9,128,128),generator=rng)
        if arm=='model_residual':
            prior=torch.linspace(-.8,.8,80).reshape(2,40)
            context=dict(expected=x-50,prior=prior,available=torch.tensor([True,False]))
            initial=model.command(x,context=context)
            info['initial_prior_max_error']=float((initial-prior).abs().max())
            assert info['initial_prior_max_error']==0
            model.head[-1].bias.fill_(10)
            changed=model.command(x,context=context)
            info['largest_correction']=float((changed-prior).abs().max())
            info['fallback_prior_max_error']=float((changed[1]-prior[1]).abs().max())
            assert info['largest_correction']<=.050001 and info['fallback_prior_max_error']==0
        else:
            info['initial_command_max_error']=float((model.command(x)-model.core.command(x)).abs().max())
            assert info['initial_command_max_error']==0
    result[arm]=info
(ROOT/'contract_check.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
