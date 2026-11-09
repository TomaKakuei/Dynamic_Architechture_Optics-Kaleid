"""Small synthetic-input contract check; no validation replay or optical scoring."""
from pathlib import Path
import json
import torch
ROOT = Path(__file__).resolve().parent
from input_network import restore, ARMS
torch.set_num_threads(1)

p = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
results = {}
for arm, spec in ARMS.items():
    torch.manual_seed(p['model_seed'])
    model, _ = restore(p['initial_checkpoint'], arm=arm, device='cpu')
    generator = torch.Generator().manual_seed(61933)
    images = model.reference[None] + 80*torch.randn((2, 9, 128, 128), generator=generator)
    with torch.no_grad():
        before = model.command(images)
        info = dict(parameters=sum(x.numel() for x in model.parameters()),
                    extra_parameters=0 if model.adapter is None else sum(x.numel() for x in model.adapter.parameters()),
                    observed_fields=spec['observed_fields'])
        if model.adapter is None:
            altered = images.clone()
            missing = [i for i in range(9) if i not in spec['observed_fields']]
            altered[:, missing] += 10000
            error = float((before-model.command(altered)).abs().max())
            assert error == 0, 'Discarded observations leaked into network'
            info['masked_input_leakage_max_error'] = error
        else:
            error = float((before-model.core.command(images)).abs().max())
            assert error == 0, 'Zero adapter changed initialization'
            info['initial_command_max_error'] = error
            if arm == 'nine_diff_adapter':
                raw = torch.nn.functional.avg_pool2d((images-model.reference)/1024, 8)
                diff = model.adapter_features(images).reshape(2, 9, 16, 16)
                restored = diff.clone()
                restored[:, :4] += diff[:, 4:5]; restored[:, 5:] += diff[:, 4:5]
                info['feature_reconstruction_max_error'] = float((restored-raw).abs().max())
                assert info['feature_reconstruction_max_error'] < 1e-6
    results[arm] = info
assert results['nine_raw_adapter']['parameters'] == results['nine_diff_adapter']['parameters']
(ROOT/'contract_check.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results))
