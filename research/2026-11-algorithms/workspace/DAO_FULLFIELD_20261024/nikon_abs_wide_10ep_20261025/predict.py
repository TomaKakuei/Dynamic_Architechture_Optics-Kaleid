"""Infer a correction from one [N,5,197] measured packet, without pose labels."""
from pathlib import Path
import argparse
import numpy as np
import torch
import train as run


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
    parser.add_argument('packet', help='Numpy file with signed wavefronts, shape [N,5,197], in waves')
    parser.add_argument('commands', help='Output Numpy file, shape [N,35]')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    raw = np.load(args.packet)
    if raw.ndim != 3 or raw.shape[1:] != (5,197) or not np.isfinite(raw).all():
        raise ValueError('Need finite [N,5,197]: ABS(0), DIFF(-.1,+.1,-.15,+.15)')
    model = load_model(args.checkpoint)
    values = []
    with torch.no_grad():
        for first in range(0,len(raw),run.MICRO):
            values.append(model.predict_packet(raw[first:first+run.MICRO]).cpu().numpy())
    np.save(args.commands,np.concatenate(values))


if __name__ == '__main__':
    main()
