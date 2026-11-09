"""Measured-input-only inference for the additive four-epoch Nikon candidate."""
from pathlib import Path
import argparse
import numpy as np
import torch
import run_trial as run


def load_model(checkpoint=None):
    checkpoint = Path(checkpoint) if checkpoint else run.HERE/'last.pt'
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    system = run.old.make_system('nikon')
    stats = dict(mean=saved['model']['extra_mean'].numpy(),std=saved['model']['extra_std'].numpy())
    model = run.make_model(system,stats)
    model.load_state_dict(saved['model'],strict=True)
    return model.eval(),system.legacy


@torch.no_grad()
def predict(model,legacy,raw,extra,maps=None,batch=128):
    raw = np.asarray(raw,dtype=np.float32)
    extra = np.asarray(extra,dtype=np.float32)
    if raw.ndim!=3 or raw.shape[1:]!=(3,49) or extra.shape!=(len(raw),run.EXTRA_DIM):
        raise ValueError('Need original raw[N,3,49], extra[N,393]')
    if maps is None:
        maps = legacy.nikon35.grid(raw)
    maps = np.asarray(maps,dtype=np.float32)
    if maps.shape!=(len(raw),3,32,32):
        raise ValueError('Original maps must have shape[N,3,32,32]')
    output=[]
    for first in range(0,len(raw),batch):
        image=torch.as_tensor(maps[first:first+batch],device='cuda')
        native=torch.as_tensor(raw[first:first+batch],device='cuda')
        added=torch.as_tensor(extra[first:first+batch],device='cuda')
        prediction=model(image,native,added)
        output.append((-(prediction*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5)).cpu().numpy())
    return np.concatenate(output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input',help='NPZ with raw, extra and optional original maps; no pose labels')
    parser.add_argument('output')
    parser.add_argument('--checkpoint',default=None)
    args=parser.parse_args()
    model,legacy=load_model(args.checkpoint)
    with np.load(args.input) as z:
        command=predict(model,legacy,z['raw'],z['extra'],z['maps'] if 'maps' in z.files else None)
    np.savez(args.output,command=command)


if __name__=='__main__':
    main()
