"""Infer a correction from one [N,7,197] measured packet, without pose labels."""
from pathlib import Path
import argparse
import importlib.util
import sys
import numpy as np
import torch
# The archived Nikon implementation imports its own module named `train`.
# Keep this entrypoint under a unique name so the legacy import stays intact.
_module_name = 'nikon_seven_packet_training'
if _module_name in sys.modules:
    run = sys.modules[_module_name]
else:
    _spec = importlib.util.spec_from_file_location(_module_name, Path(__file__).with_name('train.py'))
    run = importlib.util.module_from_spec(_spec)
    sys.modules[_module_name] = run
    _spec.loader.exec_module(run)


def load_model(checkpoint=None):
    checkpoint = Path(checkpoint) if checkpoint else run.HERE/'last.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['config']['fields_mm'] != list(run.FIELDS):
        raise ValueError('Checkpoint field order differs from the sensor packet contract')
    system = run.old.make_system('nikon')
    optics = system.legacy.NikonTensorOptics(nodes=system.legacy.nikon35.AUDIT_NODES,
                                           fields=run.FIELDS, device='cuda')
    optics = run.make_typed_optics(optics, torch.float64)
    stats = dict(mean=saved['model']['base.raw_mean'].numpy(),
                 std=saved['model']['base.raw_std'].numpy())
    model = run.make_model(system, optics, stats)
    model.load_state_dict(saved['model'], strict=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('packet', help='Numpy file with signed wavefronts, shape [N,7,197], in waves')
    parser.add_argument('commands', help='Output Numpy file, shape [N,35]')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    raw = np.load(args.packet)
    if raw.ndim != 3 or raw.shape[1:] != (run.N_FIELDS,run.NODES) or not np.isfinite(raw).all():
        raise ValueError('Need finite [N,7,197]: ABS(0), DIFF(-.05,+.05,-.1,+.1,-.15,+.15)')
    model = load_model(args.checkpoint)
    values = []
    with torch.no_grad():
        for first in range(0,len(raw),run.MICRO):
            values.append(model.predict_packet(raw[first:first+run.MICRO]).cpu().numpy())
    np.save(args.commands,np.concatenate(values))


if __name__ == '__main__':
    main()
