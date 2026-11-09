"""A single failure's camera-fidelity control; no optimization trajectory."""
from support import *
from camera_inverse import CameraModel
import json


@torch.no_grad()
def main():
    out=HERE/'debug_invalid/resolution7049.json'
    if out.exists():print(out.read_text());return
    case=7049
    with np.load(SOURCE/'training/selected_validation.npz') as z:
        slot=int(np.where(z['index']==case)[0][0]);q=z['initial'][slot].copy();prior=-z['command'][slot].copy()
    saved=json.loads((HERE/'evaluation500'/f'case_{case}.json').read_text());final=-np.array(saved['command'])
    with np.load(SOURCE/'data'/f'batch_{case//25:04d}.npz') as z:
        image=torch.as_tensor(z['speckles'][case%25].copy(),device='cuda').float()
    sigma=sigma_from_adu(image);camera=CameraModel(n=2048);results={}
    for name,x in [('prior',prior),('saved_final',final),('true_pose_diagnostic_only',q)]:
        expected,_=camera.evaluate(x)
        perfield=(((expected-image)/sigma)**2).mean((1,2))
        results[name]=dict(total_cost=float(perfield.sum()),per_field=perfield.tolist())
        print(json.dumps({name:results[name]}),flush=True)
    results['definition']='Original2048pupil/4096FFT, same known camera; true pose only diagnoses forward mismatch; no command selected from these scores'
    common.atomic(out,results)


if __name__=='__main__':main()
