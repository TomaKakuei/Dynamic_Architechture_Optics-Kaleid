"""Load a widened Nikon checkpoint and predict from one raw observation packet."""
from pathlib import Path
import sys,argparse,json
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'nikon_short_training'))
import numpy as np
import torch
import experiment as exp
from exact_refinement import ExactResponseNetwork
from fp32_response import make_typed_optics


class CandidatePredictor:
    def __init__(self,checkpoint,device='cuda'):
        self.checkpoint=Path(checkpoint).resolve();self.device=torch.device(device)
        saved=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        state=saved['model']
        self.legacy=exp.load_module('candidate_inference_nikon_job',exp.ROOT/'nikon_coordinated_center_20261005/job.py')
        base=self.legacy.CoordinatedNikon(self.legacy.previous.payloads('ordinary'),state['base.raw_mean'],state['base.raw_std'])
        model=exp.LargerNetwork(base,'nikon')
        widths=[]
        for i,layer in enumerate(model.layers):
            width=state[f'layers.{i}.linear1.weight'].shape[0];widths.append(width)
            layer.linear1=torch.nn.Linear(256,width)
            layer.linear2=torch.nn.Linear(width,256)
        native=self.legacy.NikonTensorOptics(nodes=self.legacy.nikon35.NODES,
            fields=self.legacy.nikon35.base.corrected.CONTROL_FIELDS_MM,device=self.device)
        model.__class__=ExactResponseNetwork
        model.native_optics=make_typed_optics(native,torch.float64)
        model.load_state_dict(state,strict=True)
        # Preserve parameter flags and validation's no_grad path. The new
        # trajectory-backward implementation is not needed for inference.
        self.model=model.to(self.device).eval()
        parameters=sum(p.numel() for p in model.parameters())
        assert parameters==saved['config']['parameters']
        self.metadata=dict(checkpoint=str(self.checkpoint),epoch=int(saved['epoch']),parameters=parameters,
            ffn_widths=widths,device=str(self.device),neural_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
            input_keys=['raw','optional maps'],hidden_pose_input=False,measurement_acquisitions=1,
            model_only_refinements=4,measured_feedback=False,training_backward_used_at_inference=False)

    @torch.no_grad()
    def predict(self,raw,maps=None,batch=128):
        raw=np.asarray(raw,dtype=np.float32)
        if raw.ndim==2:raw=raw[None]
        if raw.ndim!=3 or raw.shape[1:]!=(3,49) or not len(raw) or not np.isfinite(raw).all():
            raise ValueError('raw must contain finite[N,3,49] original DIFF observations')
        if maps is None:maps=self.legacy.nikon35.grid(raw)
        maps=np.asarray(maps,dtype=np.float32)
        if maps.ndim==3:maps=maps[None]
        if maps.shape!=(len(raw),3,32,32) or not np.isfinite(maps).all():
            raise ValueError('maps must contain finite[N,3,32,32] original interpolation')
        if batch<1:raise ValueError('batch must be positive')
        commands=[]
        for start in range(0,len(raw),batch):
            image=torch.as_tensor(maps[start:start+batch],device=self.device)
            measured=torch.as_tensor(raw[start:start+batch],device=self.device)
            prediction=self.model(image,measured)
            command=-(prediction*self.model.base.target_std+self.model.base.target_mean).clamp(-1.5,1.5)
            commands.append(command.cpu().numpy())
        result=np.concatenate(commands)
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite command')
        return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--batch',type=int,default=128)
    args=parser.parse_args();predictor=CandidatePredictor(args.checkpoint,args.device)
    with np.load(args.input,allow_pickle=False) as z:
        # Intentionally read no targets, hidden poses, scores or case labels.
        raw=z['raw'].copy();maps=z['maps'].copy() if 'maps' in z.files else None
    command=predictor.predict(raw,maps,args.batch)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,command=command)
    metadata=predictor.metadata|dict(count=len(command),output='Normalized35DOF actuator commands')
    exp.atomic(args.output.with_suffix('.json'),metadata)
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
