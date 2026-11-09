"""Exercise the label-free CLI and record parameter-only training changes."""
from pathlib import Path
import argparse,json,subprocess,sys
import numpy as np

HERE=Path(__file__).resolve().parent

def prepare():
    folder=HERE/'exact_local_response'
    manifest=json.loads((folder/'data_manifest.json').read_text())
    with np.load(folder/'baseline_validation.npz') as z:
        wanted=z['indices'][:128].copy();seeds=z['original_ids'][:128].copy()
    raw=[];images=[];found=[];offset=0
    for filename in manifest['files']:
        with np.load(filename) as z:
            n=len(z['seeds']);local=wanted[(wanted>=offset)&(wanted<offset+n)]-offset
            if len(local):
                raw.append(z['raw'][local].astype(np.float32))
                images.append(z['maps'][local].astype(np.float32))
                found.extend(z['seeds'][local].tolist())
        offset+=n
        if len(found)==128:break
    assert np.array_equal(found,seeds)
    np.savez_compressed(HERE/'inference_smoke_input.npz',raw=np.concatenate(raw),maps=np.concatenate(images))
    print('Prepared128original validation measurements with raw/maps only',flush=True)

def verify():
    from short_train import TrialSystem,torch,exp
    prepare()
    command=[sys.executable,'-u',str(HERE/'predict_nikon.py'),'--input',str(HERE/'inference_smoke_input.npz'),
        '--output',str(HERE/'inference_smoke_output.npz'),'--idea','exact_local_response','--batch','128']
    subprocess.run(command,check=True)
    with np.load(HERE/'inference_smoke_input.npz') as z:input_keys=z.files
    with np.load(HERE/'inference_smoke_output.npz') as z:actual=z['command']
    with np.load(HERE/'exact_local_response/selected_validation.npz') as z:expected=z['command'][:128]
    maximum=float(np.max(np.abs(actual-expected)))
    assert set(input_keys)=={'raw','maps'} and maximum<1e-6
    parameters={}
    for idea in ['closed_loop_training','nonlinear_local_model','exact_local_response']:
        system=TrialSystem(idea);model=system.model('large')
        names=dict(model.named_parameters())
        initial=system.saved['model']
        last=torch.load(HERE/idea/'last.pt',map_location='cpu',weights_only=False)['model']
        selected=torch.load(HERE/idea/'candidate.pt',map_location='cpu',weights_only=False)
        resume=torch.load(HERE/idea/'resume.pt',map_location='cpu',weights_only=False)
        optimizer_steps=sorted({int(v['step']) for v in resume['optimizer']['state'].values() if 'step' in v})
        changes={key:float((last[key].double()-initial[key].double()).square().sum()) for key in names}
        selected_delta=sum(float((selected['model'][key].double()-initial[key].double()).square().sum()) for key in names)
        row=dict(parameter_count=sum(v.numel() for v in names.values()),parameter_tensors=len(names),
            changed_parameter_tensors=sum(v>0 for v in changes.values()),
            last_parameter_delta_l2=float(np.sqrt(sum(changes.values()))),
            selected_parameter_delta_l2=float(np.sqrt(selected_delta)),selected_epoch=int(selected['epoch']),
            dtypes=sorted({str(v.dtype) for v in names.values()}),all_trainable=all(v.requires_grad for v in names.values()),
            excludes_buffers=True,source_checkpoint=str(system.warm))
        row.update(saved_optimizer_step_counts=optimizer_steps,
            optimizer_parameter_tensors=len(resume['optimizer']['state']))
        assert row['parameter_count']==7121929 and row['dtypes']==['torch.float32']
        assert row['all_trainable'] and row['last_parameter_delta_l2']>0
        assert optimizer_steps==[752]
        parameters[idea]=row
        del model,system,resume
    exp.atomic(HERE/'inference_and_parameter_check.json',dict(updated=exp.stamp(),
        cli_command=command,label_free_input_keys=input_keys,count=128,max_command_difference=maximum,
        comparison='First128original validation measurements; same128batch and selected checkpoint',
        passed=True,parameters=parameters))
    print(json.dumps(dict(passed=True,max_command_difference=maximum,parameters=parameters),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    prepare() if a.prepare_only else verify()
