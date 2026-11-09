"""Resumable original-camera predictions from the frozen full-data prior only."""
from pathlib import Path
import sys,os,json,traceback,msvcrt
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'model_residual'
OUT.mkdir(exist_ok=True)
CACHE=OUT/'camera_cache'
CACHE.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'))
sys.path.insert(0,str(ROOT.parent))
from common import np,clock,stamp,awake
from durable_io import atomic_json,retry_io
from network import restore
from camera_context import ForwardCamera
import torch
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def state(stage,**kw):
    try:atomic_json(OUT/'cache_state.json',dict(stage=stage,pid=os.getpid(),updated=stamp(),**kw))
    except PermissionError as exc:print(f'Telemetry deferred: {exc!r}',flush=True)

def stopped():
    return (ROOT/'STOP_REQUESTED.json').exists() or (OUT/'STOP_REQUESTED.json').exists()

def mmap(name,dtype,shape,initial=None):
    path=CACHE/(name+'.npy')
    if path.exists():
        a=np.load(path,mmap_mode='r+')
        if a.shape!=shape or a.dtype!=np.dtype(dtype):raise RuntimeError(f'Incompatible cache: {name}')
        return a
    a=np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
    if initial is not None:a[:]=initial
    a.flush()
    return a

def main():
    if (OUT/'cache_complete.json').exists():print('Camera context already complete.');return
    if stopped():state('stopped_before_start');return
    protocol=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'selection.npz') as z:train=z['train'].copy();validation=z['indices'].copy()
    ids=np.r_[train,validation].astype(np.int64)
    assert len(train)==27000 and len(validation)==200 and len(np.unique(ids))==len(ids)
    images=np.load(Path(protocol['source_data'])/'cache/images.npy',mmap_mode='r')
    state('opening_resumable_cache',total=len(ids),format='float32 memory maps')
    n=len(ids)
    saved_ids=mmap('ids',np.int64,(n,),ids)
    if not np.array_equal(saved_ids,ids):raise RuntimeError('Cache case order changed')
    lookup_values=np.full(len(images),-1,np.int64);lookup_values[ids]=np.arange(n)
    lookup=mmap('lookup',np.int64,(len(images),),lookup_values)
    if not np.array_equal(lookup,lookup_values):raise RuntimeError('Cache lookup changed')
    arrays={
        'expected':mmap('expected',np.float32,(n,9,128,128)),
        'prior':mmap('prior',np.float32,(n,40)),
        'available':mmap('available',np.bool_,(n,),False),
        'done':mmap('done',np.bool_,(n,),False),
        'prior_done':mmap('prior_done',np.bool_,(n,),False),
        'seconds':mmap('seconds',np.float64,(n,),0.),
        'grids':mmap('grids',np.int16,(n,9),0),
    }
    atomic_json(CACHE/'layout.json',dict(count=n,train=len(train),validation=len(validation),
        initial_checkpoint=protocol['initial_checkpoint'],schema=1,commit_rows=25,
        expected_bytes=int(arrays['expected'].nbytes)))
    prior=arrays['prior'];prior_done=arrays['prior_done'];done=arrays['done']
    if (done & ~prior_done).any():raise RuntimeError('Committed render without committed prior')
    with np.load(ROOT/'baseline200.npz') as z:
        assert np.array_equal(z['indices'],validation)
        val_prior=z['command'].astype(np.float32)
    if prior_done[len(train):].all():
        assert np.array_equal(prior[len(train):],val_prior)
    else:
        prior[len(train):]=val_prior;prior.flush()
        prior_done[len(train):]=True;prior_done.flush()
    pending=np.flatnonzero(~prior_done[:len(train)])
    if len(pending):
        state('initial_neural_predictions',completed=int(prior_done[:len(train)].sum()),total=len(train),validation_commands='reused')
        model,_=restore(protocol['initial_checkpoint'])
        model.eval()
        with torch.no_grad():
            for first in range(0,len(pending),256):
                if stopped():state('stopped_during_prior',completed=int(prior_done.sum()),total=n);return
                chunk=pending[first:first+256]
                for micro in range(0,len(chunk),16):
                    ix=chunk[micro:micro+16]
                    batch=torch.tensor(images[ids[ix]].astype(np.float32),device='cuda')
                    prediction=model.command(batch).cpu().numpy()
                    if not np.isfinite(prediction).all():raise RuntimeError('Nonfinite frozen prior')
                    prior[ix]=prediction
                prior.flush();prior_done[chunk]=True;prior_done.flush()
                completed=int(prior_done[:len(train)].sum())
                state('initial_neural_predictions',completed=completed,total=len(train),validation_commands='reused')
                if completed%2048==0 or first+len(chunk)==len(pending):
                    print(json.dumps(dict(stage='frozen_prior',completed=completed,total=len(train))),flush=True)
        del model,batch;torch.cuda.empty_cache()
    assert prior_done.all()
    camera=ForwardCamera()
    errors_path=CACHE/'render_errors.json'
    errors=json.loads(errors_path.read_text(encoding='utf-8')) if errors_path.exists() else {}
    pending_commit=[]
    render_start=clock();start_completed=int(done.sum())
    def commit():
        if not pending_commit:return
        # Publish done only after image and diagnostics have been durably flushed.
        for key in ('expected','available','seconds','grids'):retry_io(arrays[key].flush)
        atomic_json(errors_path,errors)
        done[pending_commit]=True;retry_io(done.flush)
        pending_commit.clear()
        completed=int(done.sum())
        newly=completed-start_completed
        elapsed=clock()-render_start
        mean=elapsed/newly if newly else 0.
        info=dict(completed=completed,total=n,fallbacks=int((done & ~arrays['available']).sum()),
            mean_seconds_per_packet=mean,estimated_remaining_seconds=(n-completed)*mean,
            cumulative_render_seconds=float(arrays['seconds'][done].sum()),session_seconds=elapsed,
            committed=True)
        state('rendering_model_context',**info);print(json.dumps(info),flush=True)
    state('rendering_model_context',completed=int(done.sum()),total=n,resumed=bool(done.any()))
    for row in np.flatnonzero(~done):
        if stopped():
            commit();state('stopped',completed=int(done.sum()),total=n);return
        started=clock()
        expected,available,diagnostic=camera.expected_or_fallback(-prior[row].astype(np.float64))
        if not np.isfinite(expected).all():raise RuntimeError('Nonfinite predicted camera image')
        arrays['expected'][row]=expected
        arrays['seconds'][row]=clock()-started
        arrays['available'][row]=available
        arrays['grids'][row]=diagnostic.get('grids',np.zeros(9,dtype=np.int16))
        if not available:errors[str(int(ids[row]))]=diagnostic.get('error','model domain failure')
        else:errors.pop(str(int(ids[row])),None)
        pending_commit.append(int(row))
        if len(pending_commit)>=25:commit()
    commit()
    assert done.all()
    failure=[dict(case=int(ids[r]),available=False,seconds=float(arrays['seconds'][r]),
                  error=errors.get(str(int(ids[r])),'model domain failure')) for r in np.flatnonzero(~arrays['available'])]
    result=dict(complete=True,count=n,train_count=len(train),validation_count=len(validation),
        render_seconds=float(arrays['seconds'].sum()),fallbacks=failure,
        training_render_seconds=float(arrays['seconds'][:len(train)].sum()),
        validation_render_seconds=float(arrays['seconds'][len(train):].sum()),
        training_fallbacks=int((~arrays['available'][:len(train)]).sum()),
        validation_fallbacks=int((~arrays['available'][len(train):]).sum()),
        cache_format='float32 .npy memory maps; per25case durable commits',
        frozen_prior=protocol['initial_checkpoint'],
        scope='No new acquisition or true-pose input. Original expected camera at frozen predicted initial pose.',
        inference_cost='One frozen-network prediction plus native model rendering plus learned bounded correction')
    atomic_json(OUT/'cache_complete.json',result)
    state('complete',completed=n,total=n,fallbacks=len(failure))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    with (OUT/'cache.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:awake();main()
        except Exception as exc:state('failed',error=repr(exc),traceback=traceback.format_exc());raise
        finally:awake(False)
