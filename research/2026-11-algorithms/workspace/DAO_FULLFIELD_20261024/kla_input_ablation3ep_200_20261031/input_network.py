"""Input-only KLA ablations, with a matched-capacity representation control."""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0, str(SOURCE))
from network import restore as restore_core, JointNetwork

ARMS = {
    'center_only': dict(observed_fields=[4], adapter=None),
    'three_fields': dict(observed_fields=[0, 4, 8], adapter=None),
    'nine_raw_adapter': dict(observed_fields=list(range(9)), adapter='raw'),
    'nine_diff_adapter': dict(observed_fields=list(range(9)), adapter='signed_field_difference'),
}


class InputNetwork(nn.Module):
    def __init__(self, core, arm):
        super().__init__()
        self.core, self.arm = core, arm
        self.spec = ARMS[arm]
        observed = torch.zeros(1, 9, 1, 1, dtype=torch.bool)
        observed[:, self.spec['observed_fields']] = True
        self.register_buffer('observed_fields', observed)
        self.adapter = None
        if self.spec['adapter'] is not None:
            self.adapter = nn.Sequential(nn.Linear(9*16*16, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 40))
            # Both adapter variants initially reproduce the same frozen network.
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    @property
    def reference(self):
        return self.core.reference

    def observed_input(self, images):
        # Missing images are replaced by known nominal calibration, in BOTH
        # train and inference. Nothing else sees the discarded measurements.
        return torch.where(self.observed_fields, images, self.reference[None])

    def adapter_features(self, images):
        raw = (images-self.reference)/1024.
        if self.spec['adapter'] == 'signed_field_difference':
            center = raw[:, 4:5]
            raw = torch.cat((raw[:, :4]-center, center, raw[:, 5:]-center), dim=1)
        return F.avg_pool2d(raw, 8).flatten(1)

    def forward(self, images):
        images = self.observed_input(images)
        logits = self.core(images)
        if self.adapter is not None:
            logits = logits+self.adapter(self.adapter_features(images))
        return logits

    def command(self, images):
        return -torch.tanh(self(images))


def restore(path, arm=None, device='cuda'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if 'input_ablation_arm' in payload:
        saved_arm = payload['input_ablation_arm']
        if arm is not None and arm != saved_arm:
            raise ValueError('Checkpoint/input-arm mismatch')
        model = InputNetwork(JointNetwork(payload['reference']), saved_arm)
        model.load_state_dict(payload['model'])
    else:
        assert arm in ARMS
        core, _ = restore_core(path, device='cpu')
        model = InputNetwork(core, arm)
    return model.to(device).eval(), payload
