"""Nikon inference from measured raw/maps only; never reads targets or poses."""
from short_train import TrialSystem,HERE,exp,torch,np
from pathlib import Path
import argparse,json

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True,help='NPZ containing raw[N,3,49] and optional maps[N,3,32,32]')
p.add_argument('--output',type=Path,required=True,help='Output NPZ of normalized35DOF commands')
p.add_argument('--idea',default='recommended')
p.add_argument('--checkpoint',choices=['candidate','last'],default='candidate')
p.add_argument('--batch',type=int,default=128)
a=p.parse_args()
idea=a.idea
if idea=='recommended':idea=json.loads((HERE/'recommended_nikon_model.json').read_text())['idea']
system=TrialSystem(idea);model=system.model('large')
path=HERE/idea/(a.checkpoint+'.pt')
saved=torch.load(path,map_location='cpu',weights_only=False)
# Preserve validation's parameter flags; freezing them changed the executed
# numerical path slightly. no_grad below still disables backward recording.
model.load_state_dict(saved['model']);model.eval()
with np.load(a.input,allow_pickle=False) as z:
    raw=z['raw'].copy()
    if raw.ndim==2:raw=raw[None]
    images=z['maps'].copy() if 'maps' in z.files else system.legacy.nikon35.grid(raw)
    if images.ndim==3:images=images[None]
assert raw.shape[1:]==(3,49) and images.shape==(len(raw),3,32,32)
assert np.isfinite(raw).all() and np.isfinite(images).all()
commands=[]
# Match validation's no-grad context while retaining optical forward-mode AD.
with torch.no_grad():
    for start in range(0,len(raw),a.batch):
        image=torch.as_tensor(images[start:start+a.batch],dtype=torch.float32,device='cuda')
        measured=torch.as_tensor(raw[start:start+a.batch],dtype=torch.float32,device='cuda')
        prediction=model(image,measured)
        command=-(prediction*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5)
        commands.append(command.cpu().numpy())
command=np.concatenate(commands)
assert np.isfinite(command).all() and np.abs(command).max()<=1.5000001
a.output.parent.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output,command=command)
metadata=dict(idea=idea,checkpoint=str(path),checkpoint_epoch=int(saved['epoch']),count=len(command),
    inputs_used=['raw','maps or original interpolation of raw'],hidden_pose_input=False,
    network_parameter_dtypes=sorted({str(x.dtype) for x in model.parameters()}),
    inference_protocol=system.trial_config.get('inference_protocol','direct network'),
    output='Normalized35DOF actuator commands; seven bodies with dx,dy,dz,tx,ty',
    measured_feedback_executed=False)
exp.atomic(a.output.with_suffix('.json'),metadata)
print(json.dumps(metadata,indent=2))
