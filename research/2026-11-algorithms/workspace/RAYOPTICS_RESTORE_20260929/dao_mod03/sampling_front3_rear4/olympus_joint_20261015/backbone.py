from common import DAO
import torch, importlib.util
_spec = importlib.util.spec_from_file_location('olympus_canonical', DAO / 'canonical_three_phase_universal/models.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

class ModularNetwork(_module.ArchA_CanonicalResNet):
    def __init__(self, groups):
        super().__init__(9, [5] * groups)
        y, x = torch.meshgrid((torch.arange(128)+.5)/64-1, (torch.arange(128)+.5)/64-1, indexing='ij')
        basis = torch.stack([torch.ones_like(x), x, y, x*y, x*x, y*y, torch.sin(torch.pi*x), torch.sin(torch.pi*y)])
        self.register_buffer('sensor_moments', basis)
        self.head = _module.GroupFactorizedHead(256 + 72, [5] * groups)

def build(branch):
    return ModularNetwork(3 if branch == 'front' else 4)

