"""Cache F(-g(I)) once for3000train+200test; no labels used or new captures."""
from pathlib import Path
import sys,os,json,traceback
ROOT = Path(__file__).resolve().parent
OUT = ROOT/'model_residual'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'))
sys.path.insert(0,str(ROOT.parent))
from common import np,clock,stamp,awake
from durable_io import atomic_json,temporary,retry_io
from network import restore
from camera_context import ForwardCamera
import torch
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def save_npz(path,**arrays):
    tmp=temporary(path)
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    retry_io(lambda:os.replace(tmp,path))

def state(stage,**kw):
    try:atomic_json(OUT/'cache_state.json',dict(stage=stage,pid=os.getpid(),updated=stamp(),**kw))
    except PermissionError as exc:print(f'Telemetry deferred: {exc!r}',flush=True)

def main():
    if (OUT/'cache_complete.json').exists():print('Camera context already complete.');return
    protocol=json.loads((ROOT/'protocol.json').read_text())
    with np.load(ROOT/'selection.npz') as z:train=z['train'].copy();validation=z['indices'].copy()
    ids=np.r_[train,validation]
    images=np.load(Path(protocol['source_data'])/'cache/images.npy',mmap_mode='r')
    prior_path=OUT/'priors.npz'
    if prior_path.exists():
        with np.load(prior_path) as z:
            assert np.array_equal(ids,z['ids']);prior=z['prior'].copy()
    else:
        state('initial_neural_predictions',training_cases=len(train),validation_commands='reused')
        model,_=restore(protocol['initial_checkpoint'])
        with torch.no_grad():
            commands=[model.command(torch.tensor(images[train[first:first+16]].astype(np.float32),device='cuda')).cpu().numpy()
                      for first in range(0,len(train),16)]
        with np.load(ROOT/'baseline200.npz') as z:validation_commands=z['command'].copy()
        prior=np.r_[np.concatenate(commands),validation_commands]
        save_npz(prior_path,ids=ids,prior=prior)
        del model;torch.cuda.empty_cache()
    camera=ForwardCamera()
    cache=OUT/'camera_cache';cache.mkdir(exist_ok=True)
    records=[]
    for i,case in enumerate(ids):
        path=cache/f'{int(case)}.npz'
        if path.exists():
            with np.load(path) as z:
                records.append(dict(case=int(case),available=bool(z['available']),seconds=float(z['seconds']),error=str(z['error'])))
            continue
        if (ROOT/'STOP_REQUESTED.json').exists():state('stopped',completed=len(records),total=len(ids));return
        start=clock()
        expected,available,diagnostic=camera.expected_or_fallback(-prior[i].astype(np.float64))
        seconds=clock()-start
        error=diagnostic.get('error','')
        save_npz(path,expected=expected,prior=prior[i].astype(np.float32),available=np.asarray(available),
                 seconds=np.asarray(seconds),error=np.asarray(error),grids=np.asarray(diagnostic.get('grids',[])))
        records.append(dict(case=int(case),available=available,seconds=seconds,error=error))
        if (i+1)%25==0 or i==0 or i+1==len(ids):
            mean=float(np.mean([r['seconds'] for r in records[-50:]]))
            info=dict(completed=len(records),total=len(ids),fallbacks=sum(not r['available'] for r in records),
                      mean_seconds_per_packet=mean,estimated_remaining_seconds=(len(ids)-len(records))*mean)
            state('rendering_model_context',**info);print(json.dumps(info),flush=True)
    failure=[r for r in records if not r['available']]
    result=dict(complete=True,count=len(records),train_count=len(train),validation_count=len(validation),
        render_seconds=sum(r['seconds'] for r in records),fallbacks=failure,
        training_render_seconds=sum(r['seconds'] for r in records[:len(train)]),
        validation_render_seconds=sum(r['seconds'] for r in records[len(train):]),
        training_fallbacks=sum(not r['available'] for r in records[:len(train)]),
        validation_fallbacks=sum(not r['available'] for r in records[len(train):]),
        scope='No new acquisition or true-pose input. Expected camera at frozen predicted initial pose.',
        inference_cost='One frozen-network prediction plus native model rendering plus learned bounded correction')
    atomic_json(OUT/'cache_complete.json',result)
    state('complete',completed=len(records),fallbacks=len(failure))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    try:awake();main()
    except Exception as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:awake(False)
