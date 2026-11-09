"""Four real training samples on CPU while the existing GPU run continues."""
from pathlib import Path
import os,sys,json,time
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'nikon_short_training'))
import torch,numpy as np
import experiment as exp
from exact_refinement import ExactResponseNetwork
from fp32_response import make_typed_optics
from trajectory_backward import four_steps
torch.set_num_threads(1)


def main():
    tick=time.perf_counter()
    legacy=exp.load_module('cpu_probe_nikon_job',exp.ROOT/'nikon_coordinated_center_20261005/job.py')
    source=exp.ROOT/'nikon_coordinated_center_20261005'
    payload=torch.load(source/'candidate.pt',map_location='cpu',weights_only=False)
    base=legacy.CoordinatedNikon(legacy.previous.payloads('ordinary'),payload['raw_mean'],payload['raw_std'])
    model=exp.LargerNetwork(base,'nikon')
    for layer in model.layers:
        layer.linear1=torch.nn.Linear(256,4496);layer.linear2=torch.nn.Linear(4496,256)
    native=legacy.NikonTensorOptics(nodes=legacy.nikon35.NODES,
        fields=legacy.nikon35.base.corrected.CONTROL_FIELDS_MM,device='cpu')
    model.__class__=ExactResponseNetwork
    model.native_optics=make_typed_optics(native,torch.float64)
    saved=torch.load(HERE.parent/'original_routes_40ep_20261025/nikon/candidate.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model']);model.eval();del saved,payload
    probe=json.loads((HERE/'diagnosis.json').read_text())['training_indices']
    indices=probe[:2]+probe[-2:]
    sources=legacy.sources();images=[];raws=[];truths=[]
    for index in indices:
        if index<30000:
            path,_,training,_=sources[index//100];assert training
            with np.load(path) as z:
                images.append(z['maps'][index%100]);raws.append(z['raw'][index%100]);truths.append(z['targets'][index%100])
        else:
            offset=index-30000
            path=HERE.parent/'original_routes_40ep_20261025/nikon_data_v2'/f'block_{offset//100:03d}.npz'
            with np.load(path) as z:
                images.append(z['image'][offset%100]);raws.append(z['raw'][offset%100]);truths.append(z['q'][offset%100])
    image=torch.tensor(np.array(images),dtype=torch.float32)
    raw=torch.tensor(np.array(raws),dtype=torch.float32)
    target=torch.tensor(np.array(truths),dtype=torch.float64)
    optics=legacy.NikonTensorOptics(device='cpu')
    with torch.no_grad():
        neural=exp.LargerNetwork.forward(model,image,raw)
        prior=(neural*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5).double()
        measured=raw.flatten(1).double()/7.
        prediction=model(image,raw)
        physical=four_steps(prior,measured,model)
        reconstructed=((physical-model.base.target_mean)/model.base.target_std).float()
    forward_difference=float((prediction-reconstructed).abs().max())
    def objective(physical):
        metrics=optics.metrics(target-physical)
        absolute=metrics['absolute'].square().mean(-1).clamp_min(1e-20).sqrt()
        difference=metrics['difference'].square().mean(-1).clamp_min(1e-20).sqrt()
        fields=torch.cat((absolute[:,:1],difference[:,1:]),dim=1)
        assert bool(metrics['valid'].all())
        ell=torch.log1p((fields/.07).square());term=ell/(1+ell)
        pose=torch.nn.functional.smooth_l1_loss(
            (physical-model.base.target_mean.double())/model.base.target_std.double(),
            (target-model.base.target_mean.double())/model.base.target_std.double(),beta=.25)
        return (.5*term.mean(1)+.5*term.amax(1)).mean()+.02*pose
    variable=prior.clone().requires_grad_(True)
    loss=objective(four_steps(variable,measured,model))
    gradient=torch.autograd.grad(loss,variable)[0]
    rows=[];generator=torch.Generator().manual_seed(2026102522)
    for kind in ('gradient','random'):
        direction=gradient if kind=='gradient' else torch.randn(prior.shape,generator=generator,dtype=prior.dtype)
        direction=direction/direction.square().mean().sqrt()
        predicted=float((gradient*direction).sum())
        for h in (1e-4,3e-5):
            with torch.no_grad():
                minus=objective(four_steps(prior-h*direction,measured,model))
                plus=objective(four_steps(prior+h*direction,measured,model))
            observed=float((plus-minus)/(2*h))
            rows.append(dict(direction=kind,h=h,predicted=predicted,observed=observed,
                relative_error=abs(predicted-observed)/max(1e-8,abs(predicted),abs(observed)),negative_change=float(minus-loss.detach())))
    assert not torch.cuda.is_initialized(),'CPU probe must not allocate a CUDA context'
    passed=forward_difference<1e-5 and all(r['relative_error']<.002 for r in rows if r['h']==3e-5)
    result=dict(status='passed' if passed else 'failed',device='cpu',real_nikon=True,training_indices=indices,
        training_count=4,validation_cases=0,optimizer_updates=0,cuda_initialized=False,
        forward_max_difference=forward_difference,directions=rows,seconds=time.perf_counter()-tick,
        gpu16sample_check_still_required=True)
    exp.atomic(HERE/'native_cpu_check.json',result);print(json.dumps(result),flush=True)
    assert passed


if __name__=='__main__':main()
