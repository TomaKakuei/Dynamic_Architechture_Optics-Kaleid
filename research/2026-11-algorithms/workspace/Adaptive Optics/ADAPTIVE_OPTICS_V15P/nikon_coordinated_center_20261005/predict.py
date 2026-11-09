"""Kaleid-compatible Nikon raw-observation predictor for the coordinated model."""
from job import HERE,CoordinatedNikon,previous,torch,np,nikon35,json,sha256
from pathlib import Path


class Predictor:
    def __init__(self,checkpoint=None,device=None,require_ready=True):
        checkpoint=Path(checkpoint or HERE/'candidate.pt')
        if require_ready:
            manifest=json.loads((HERE/'candidate_manifest.json').read_text(encoding='utf-8'))
            if manifest['state']!='ready_for_controller_use' or sha256(checkpoint)!=manifest['sha256']:
                raise RuntimeError('Candidate completion/integrity gate failed')
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=CoordinatedNikon(previous.payloads('ordinary'),payload['raw_mean'],payload['raw_std'])
        self.model.load_state_dict(payload['state_dict']);self.model.to(self.device).eval()

    def __call__(self,raw):
        if isinstance(raw,dict):raw=raw['network']
        raw=np.asarray(raw).reshape(1,3,49)
        image=nikon35.grid(raw)
        raw=raw.astype(np.float32)
        with torch.inference_mode():
            prediction=self.model.physical_prediction(torch.tensor(image,device=self.device),torch.tensor(raw,device=self.device))
        return prediction[0].cpu().numpy()


_PREDICTOR=None


def nikon_predict(raw,architecture='arch_a'):
    global _PREDICTOR
    if architecture not in ('arch_a','coordinated'):raise ValueError(architecture)
    if _PREDICTOR is None:_PREDICTOR=Predictor()
    return _PREDICTOR(raw)


def nikon_calibration():
    return tuple(np.load(previous.SOURCE/'results'/f'nikon_{name}_calibration35.npz')['jacobian'] for name in ('center','capture'))
