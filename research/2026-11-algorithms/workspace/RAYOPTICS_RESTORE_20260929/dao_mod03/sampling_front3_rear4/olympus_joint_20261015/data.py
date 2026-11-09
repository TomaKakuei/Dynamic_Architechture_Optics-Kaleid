"""Reuse old blocks and poses; render only missing whole-assembly observations."""
from common import *
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import threading, importlib.util, gc

_spec=importlib.util.spec_from_file_location('olympus_relay_source',BASE/'camera.py')
_camera_module=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_camera_module)
RelayCamera=_camera_module.RelayCamera

def sources():
    rows=read(PREVIOUS/'results/dataset_manifest.json')['records']
    result={b:[] for b in BRANCHES}
    for row in rows:
        result[row['branch']].append(dict(batch=int(row['batch']),path=str(relocate(row['path']))))
    for b in BRANCHES:
        result[b].sort(key=lambda r:r['batch'])
    return result

def load_poses(records):
    cache=HERE/'source_poses.npz'
    if cache.exists():
        with np.load(cache) as z:return {b:z[b].copy() for b in BRANCHES}
    poses={b:np.empty((TOTAL,35),np.float64) for b in BRANCHES}
    work=[(b,row) for b in BRANCHES for row in records[b]]
    def fetch(item):
        branch,row=item
        with np.load(row['path']) as z:
            return branch,row['batch'],z['sample_index'].astype(int),z['targets_normalized'].astype(np.float64)
    state('reading_reused_poses',completed_blocks=0,total_blocks=len(work),io_workers=8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number,(branch,batch,ids,q) in enumerate(pool.map(fetch,work),1):
            poses[branch][ids]=q
            if number%25==0:state('reading_reused_poses',completed_blocks=number,total_blocks=len(work),io_workers=8)
    np.savez(cache,**poses)
    return poses

class Blocks:
    def __init__(self, capacity=4):
        self.cache=OrderedDict()
        self.capacity=capacity
        self.records=sources()
    def read(self, kind, batch):
        key=(kind,int(batch))
        if key not in self.cache:
            path=HERE/'data'/f'batch_{batch:04d}.npz' if kind=='joint' else Path(self.records[kind][batch]['path'])
            with np.load(path) as z:
                self.cache[key]=(z['speckles'].copy(),z['targets_normalized'].astype(np.float64))
            if len(self.cache)>self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

_thread_state=threading.local()
def representation(task):
    index, poses = task
    if not hasattr(_thread_state,'cameras'):
        _thread_state.cameras={24:RelayCamera(device='cpu',degree=24)}
    cameras=_thread_state.cameras
    primary=cameras[24]
    first=0 if index<TRAIN else TRAIN
    count=TRAIN if index<TRAIN else VALIDATION
    local=index-first
    rejections=[]
    for attempt in range(200):
        rear_index=first+(local+attempt*685)%count
        q=poses['front'][index]+poses['rear'][rear_index]
        if not primary.engine.geometry.path(q)['safe']:
            rejections.append('mechanics');continue
        try:
            selected=None
            for degree in (24,28,32,36):
                if degree not in cameras:
                    # At most two per-thread polynomial fits remain in memory.
                    for old in tuple(cameras):
                        if old!=24:del cameras[old]
                    cameras[degree]=RelayCamera(device='cpu',degree=degree)
                rows=cameras[degree].wave_representation(q,verify=False)
                if max(r['fit_rms_waves'] for r in rows)<=.0003:
                    selected=degree;break
            if selected is None:
                raise RuntimeError(f'Joint sample {index}: phase representation needs refinement beyond degree36')
        except ValueError as exc:
            rejections.append(str(exc));continue
        return dict(index=index,q=q,rows=rows,degree=selected,rear_index=rear_index,rejections=rejections)
    raise RuntimeError(f'Cannot form joint observation {index}; rejected {len(rejections)} physical proposals')

def render(camera, row):
    import torch
    degree=row['degree']
    resource=camera.resources(2048)
    if resource['v'].shape[1]!=degree+1:
        resource['v']=torch.as_tensor(np.polynomial.chebyshev.chebvander(resource['coords'],degree),
                                      dtype=torch.float64,device='cuda')
    images=[]
    rng=np.random.get_state()
    np.random.seed((SEED+row['index']*1031)%2**32)
    try:
        for field in row['rows']:
            intensity=camera.integrate_field(field,2048)
            images.append(camera.engine.camera_noise(intensity).astype(np.uint16))
    finally:np.random.set_state(rng)
    return np.stack(images)

def generate(workers=8, limit=None):
    import torch, psutil
    (HERE/'data').mkdir(exist_ok=True)
    records=sources()
    poses=load_poses(records)
    batches=list(range(TOTAL//25))
    missing=[b for b in batches if not (HERE/'data'/f'batch_{b:04d}.npz').exists()]
    if limit is not None:missing=missing[:limit]
    if not missing:return
    # Threads share the Python runtime; bounded futures avoid multiplying Torch RAM.
    available=psutil.virtual_memory().available/2**30
    workers=max(2,min(workers,int(max(2,(available-.65)/.12))))
    state('generating_joint_observations',completed=(TOTAL//25-len([b for b in batches if not (HERE/'data'/f'batch_{b:04d}.npz').exists()]))*25,
          target=TOTAL,cpu_workers=workers)
    start=clock()
    camera=RelayCamera(device='cuda')
    targets=[i for b in missing for i in range(b*25,(b+1)*25)]
    pending=OrderedDict()
    iterator=iter(targets)
    processed=0;buffer=[];block_start=clock()
    with ThreadPoolExecutor(max_workers=workers,thread_name_prefix='phase') as pool:
        for _ in range(workers):
            i=next(iterator,None)
            if i is not None:pending[i]=pool.submit(representation,(i,poses))
        while pending:
            i,future=pending.popitem(last=False)
            row=future.result()
            j=next(iterator,None)
            if j is not None:pending[j]=pool.submit(representation,(j,poses))
            images=render(camera,row)
            buffer.append((row,images))
            processed+=1
            if processed==1 or processed%5==0:
                elapsed=clock()-start
                completed=len(list((HERE/'data').glob('batch_*.npz')))*25+len(buffer)
                state('generating_joint_observations',completed=completed,target=TOTAL,cpu_workers=workers,
                      run_samples=processed,active_seconds=elapsed,seconds_per_sample=elapsed/processed,
                      remaining_generation_seconds=(TOTAL-completed)*elapsed/processed,
                      ram_available_gib=psutil.virtual_memory().available/2**30)
            if len(buffer)==25:
                batch=i//25
                output=HERE/'data'/f'batch_{batch:04d}.npz'
                temporary=output.with_suffix('.partial')
                with temporary.open('wb') as stream:
                    np.savez(stream,speckles=np.stack([x for r,x in buffer]),
                             targets_normalized=np.stack([r['q'] for r,x in buffer]).astype(np.float32),
                             sample_index=np.array([r['index'] for r,x in buffer]),
                             rear_source_index=np.array([r['rear_index'] for r,x in buffer]),
                             phase_fit_degree=np.array([r['degree'] for r,x in buffer]),
                             phase_fit_rms=np.array([[f['fit_rms_waves'] for f in r['rows']] for r,x in buffer]))
                commit(temporary,output)
                atomic(output.with_suffix('.json'),dict(batch=batch,count=25,active_seconds=clock()-block_start,
                       source_front_indices=[r['index'] for r,x in buffer],source_rear_indices=[r['rear_index'] for r,x in buffer],
                       rejections=[r['rejections'] for r,x in buffer],independent_replay=False))
                print('JOINT_DATA', (batch+1)*25, 'of', TOTAL, 'seconds', round(clock()-start,2),flush=True)
                buffer=[];block_start=clock()
    del camera,poses
    gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--limit',type=int)
    a=parser.parse_args()
    generate(a.workers,a.limit)



