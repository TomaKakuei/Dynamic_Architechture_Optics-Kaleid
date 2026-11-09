"""CPU-only command parity check on four already saved validation predictions."""
from predict_candidate import *


def main():
    out=HERE/'trajectory_4ep'
    predictor=CandidatePredictor(out/'candidate.pt',device='cpu')
    epoch=predictor.metadata['epoch']
    path=out/('baseline_validation.npz' if epoch==0 else f'validation_{epoch:03d}.npz')
    positions=np.array([0,1999,3999,5999])
    with np.load(path) as z:
        indices=z['indices'][positions].copy();expected=z['command'][positions].copy()
    # Only observation arrays are supplied to the predictor; truth is not read.
    files=predictor.legacy.sources();raw=[];maps=[]
    for index in indices:
        source=files[index//100][0]
        with np.load(source) as z:
            raw.append(z['raw'][index%100]);maps.append(z['maps'][index%100])
    command=predictor.predict(np.array(raw),np.array(maps),batch=4)
    difference=float(np.max(np.abs(command-expected)))
    raw_only=predictor.predict(np.array(raw),batch=4)
    raw_only_difference=float(np.max(np.abs(raw_only-expected)))
    passed=np.allclose(command,expected,atol=5e-5,rtol=1e-4) and np.allclose(raw_only,expected,atol=5e-5,rtol=1e-4)
    assert not torch.cuda.is_initialized()
    report=dict(status='passed' if passed else 'failed',epoch=epoch,count=4,device='cpu',
        cuda_context_used=False,targets_or_hidden_pose_read=False,optimizer_updates=0,
        max_abs_command_difference=difference,raw_only_max_abs_command_difference=raw_only_difference,metadata=predictor.metadata,
        scope='CPU standalone loader versus saved GPU validation commands, four inputs only; no new recovery estimate')
    exp.atomic(HERE/'prediction_entry_check.json',report);print(json.dumps(report),flush=True)
    assert passed


if __name__=='__main__':main()
