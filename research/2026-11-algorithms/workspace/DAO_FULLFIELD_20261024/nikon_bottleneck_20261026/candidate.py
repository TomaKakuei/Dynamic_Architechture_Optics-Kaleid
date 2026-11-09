"""Single-acquisition candidate: original4 native steps +4 full-packet steps.
No target/pose labels enter these functions. Frozen prior-supervision E4.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'nikon_sevenbody_diagnosis_20261025'))
import sensor_control as sc
run,np,torch=sc.run,sc.np,sc.torch
CHECKPOINT=ROOT/'nikon_prior_supervision_4ep_20261025/last.pt'


def load():
    system=run.old.make_system('nikon');system.__class__=run.AdditiveSystem
    p=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model']
    stats=dict(mean=p['extra_mean'].numpy(),std=p['extra_std'].numpy())
    model=run.make_model(system,stats);model.load_state_dict(p,strict=True);model.eval()
    return model,system,sc.AllSensors(system)


def prior(model,image,raw,extra):
    normalized=run.exp.LargerNetwork.forward(model,image,raw)
    normalized=normalized+model.extra_branch((extra-model.extra_mean)/model.extra_std)
    return (normalized*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5).double()


def refine(command4,initial_prior,raw,extra,optical):
    from trajectory_backward import TrajectoryStep
    measured=torch.cat((raw.flatten(1).double()/7.,extra[:,:197].double()/np.sqrt(197),extra[:,197:].double()/7.),1)
    physical=-command4.double()
    for _ in range(4):physical=TrajectoryStep.apply(physical,initial_prior,measured,optical)
    return -physical


@torch.no_grad()
def predict(model,optical,image,raw,extra):
    initial_prior=prior(model,image,raw,extra)
    physical=run.four_steps(initial_prior,raw.flatten(1).double()/7.,model)
    # Preserve the original4step deployed output conversion before extension.
    normalized=((physical-model.base.target_mean)/model.base.target_std).float()
    command4=-(normalized*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5)
    return refine(command4,initial_prior,raw,extra,optical)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
    model,system,optical=load()
    with np.load(a.input) as z:
        raw=np.asarray(z['raw'],dtype=np.float32);extra=np.asarray(z['extra'],dtype=np.float32)
        image=z['maps'] if 'maps' in z.files else system.legacy.nikon35.grid(raw)
    commands=[]
    for start in range(0,len(raw),128):
        tensors=[torch.as_tensor(x[start:start+128],device='cuda',dtype=torch.float32) for x in (image,raw,extra)]
        commands.append(predict(model,optical,*tensors).cpu().numpy())
    np.savez(a.output,command=np.concatenate(commands))
