"""One coordinated Nikon candidate, native noisy inputs, 50 center-loss epochs."""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'
import sys, json, math, time, csv, argparse, traceback, ctypes, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
PREVIOUS=ROOT/'expanded_differentiable_20261005'
sys.path.insert(0,str(PREVIOUS))
import nikon_job as previous
sys.path.insert(0,str(HERE))
np,torch,nikon35=previous.np,previous.torch,previous.nikon35
atomic_json,sha256,save_torch=previous.atomic_json,previous.sha256,previous.save_torch
GROUPS=previous.GROUPS
SOURCE=previous.SOURCE
NikonTensorOptics=previous.NikonTensorOptics
EPOCHS=50
SEED=202610051
INITIAL=PREVIOUS/'nikon/epochs/epoch_029.pt'
FILES=(Path(__file__),HERE/'predict.py')


def state(status,**kw):
    atomic_json(HERE/'state.json',dict(status=status,pid=os.getpid(),updated_unix=time.time(),**kw))


def sources():
    out=[]
    for gi,g in enumerate(GROUPS):
        for number in range(100):
            path=(SOURCE/'data'/f'nikon_{g}_joint_context' if number<50 else PREVIOUS/'data_supplement'/g)/f'batch_{number:04d}.npz'
            training=number<40 or 50<=number<90
            out.append((path,gi,training,number<50))
    return out


def read_file(item):
    path,gi,training,old=item
    with np.load(path) as d:
        x=d['maps'].astype(np.float32);raw=d['raw'].astype(np.float32)
        y=d['targets'].astype(np.float32);seeds=d['seeds'].copy()
        if x.shape!=(100,3,32,32) or raw.shape!=(100,3,49) or y.shape!=(100,35):raise RuntimeError(f'Wrong shapes {path}')
        if str(d['sampling_mode'])!='joint_context' or not np.isfinite(x).all() or not np.isfinite(raw).all():raise RuntimeError(f'Invalid data {path}')
    row=(x,raw,y,seeds,np.full(100,gi),np.full(100,old))
    return training,row,dict(path=str(path.resolve()),sha256=sha256(path))


def load_data():
    state('loading_existing_noisy_raw_and_maps')
    rows={True:[],False:[]};fingerprints=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for training,row,fp in pool.map(read_file,sources()):
            rows[training].append(row);fingerprints.append(fp)
    combine=lambda items:tuple(np.concatenate([r[k] for r in items]) for k in range(6))
    train,val=combine(rows[True]),combine(rows[False])
    if len(train[0])!=24000 or len(val[0])!=6000:raise RuntimeError('Sample counts changed')
    if len(np.unique(train[3]))!=24000 or len(np.unique(val[3]))!=6000 or np.intersect1d(train[3],val[3]).size:raise RuntimeError('Seed overlap')
    old_split=json.loads((PREVIOUS/'nikon/split_manifest.json').read_text(encoding='utf-8'))
    if not np.array_equal(val[3],old_split['validation_seeds']):raise RuntimeError('Validation order changed')
    indices=np.asarray(old_split['train_probe_indices'])
    if not np.array_equal(train[3][indices],old_split['train_probe_seeds']):raise RuntimeError('Training probe changed')
    probe=tuple(a[indices] for a in train)
    return train,val,probe,fingerprints


class CoordinatedNikon(torch.nn.Module):
    """Original encoders plus native-node attention and cross-module residuals.

    All three module tokens attend to the SAME noisy node stack and to each
    other. The final zero-initialized heads preserve the archived model exactly
    at initialization. Seven 5-DOF bodies remain grouped as15/15/5 outputs.
    """
    def __init__(self,payloads,raw_mean,raw_std):
        super().__init__()
        self.base=previous.JointModel(payloads)
        self.register_buffer('raw_mean',torch.as_tensor(raw_mean,dtype=torch.float32))
        self.register_buffer('raw_std',torch.as_tensor(raw_std,dtype=torch.float32).clamp_min(.015))
        self.register_buffer('target_mean',torch.tensor(np.concatenate([p['target_mean'] for p in payloads]),dtype=torch.float32))
        self.register_buffer('target_std',torch.tensor(np.concatenate([p['target_std'] for p in payloads]),dtype=torch.float32))
        xy=np.tile(np.asarray(nikon35.NODES,dtype=np.float32),(3,1))
        field=np.repeat([0.,-1.,1.],49)[:,None]
        coords=np.c_[xy,field,(xy*xy).sum(1)]
        self.register_buffer('node_coordinates',torch.tensor(coords,dtype=torch.float32))
        width=128
        self.node_encoder=torch.nn.Sequential(torch.nn.Linear(5,64),torch.nn.GELU(),torch.nn.Linear(64,width),torch.nn.GELU())
        self.raw_global=torch.nn.Sequential(torch.nn.Linear(147,width),torch.nn.GELU(),torch.nn.Linear(width,width))
        self.feature_projections=torch.nn.ModuleList([torch.nn.Linear(256,width) for _ in GROUPS])
        self.pose_projections=torch.nn.ModuleList([torch.nn.Linear(n,width) for n in (15,15,5)])
        self.module_identity=torch.nn.Parameter(torch.zeros(1,3,width))
        self.cross_norm=torch.nn.LayerNorm(width)
        self.node_attention=torch.nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.module_norm=torch.nn.LayerNorm(width)
        self.module_attention=torch.nn.MultiheadAttention(width,4,dropout=0.,batch_first=True)
        self.ff_norm=torch.nn.LayerNorm(width)
        self.ff=torch.nn.Sequential(torch.nn.Linear(width,256),torch.nn.GELU(),torch.nn.Linear(256,width))
        self.correction_heads=torch.nn.ModuleList([torch.nn.Linear(width,n) for n in (15,15,5)])
        for head in self.correction_heads:
            torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)

    def forward(self,image,raw):
        features=[];estimates=[]
        for network in self.base.networks:
            f=network.stage3(network.stage2(network.stage1(network.stem(image))))
            f=network.pool(f).flatten(1)
            features.append(f);estimates.append(network.head(f))
        native=((raw-self.raw_mean)/self.raw_std).flatten(1)
        coordinates=self.node_coordinates[None].expand(len(image),-1,-1)
        nodes=self.node_encoder(torch.cat((native[:,:,None],coordinates),dim=-1))
        context=torch.stack([fp(f)+pp(p) for fp,pp,f,p in zip(self.feature_projections,self.pose_projections,features,estimates)],dim=1)
        context=context+self.raw_global(native)[:,None]+self.module_identity
        context=context+self.node_attention(self.cross_norm(context),nodes,nodes,need_weights=False)[0]
        normalized=self.module_norm(context)
        context=context+self.module_attention(normalized,normalized,normalized,need_weights=False)[0]
        context=context+self.ff(self.ff_norm(context))
        correction=torch.cat([head(context[:,i]) for i,head in enumerate(self.correction_heads)],dim=-1)
        return torch.cat(estimates,dim=-1)+correction

    def physical_prediction(self,image,raw):
        return self(image,raw)*self.target_std+self.target_mean


class CenterObjective:
    def __init__(self,model,device='cuda'):
        self.mean=model.target_mean;self.std=model.target_std
        self.optics=NikonTensorOptics(device=device)

    def corrected(self,prediction,target):
        command=-(prediction*self.std+self.mean).clamp(-1.5,1.5)
        return target.to(torch.float64)+command.to(torch.float64)

    def __call__(self,prediction,target):
        q=self.corrected(prediction,target)
        with torch.no_grad():valid=self.optics.metrics(q)['valid']
        if bool(valid.any()):
            metric=self.optics.metrics(q if bool(valid.all()) else q[valid])
            relative=metric['center']/.07
            central=torch.sqrt(1.+relative.square())-1.
            excess=.5*torch.relu(relative-1.).square()
            physical=(central.sum()+excess.sum()+100.*(~valid).sum())/len(q)
        else:
            physical=prediction.sum()*0.+100.
        pose=torch.nn.functional.smooth_l1_loss(prediction,(target-self.mean)/self.std,beta=.25)
        loss=physical+.02*pose
        return loss,dict(physical_loss=physical,pose_loss=pose,invalid_count=int((~valid).sum()))


def fresh_model(raw_mean,raw_std,device='cuda'):
    model=CoordinatedNikon(previous.payloads('ordinary'),raw_mean,raw_std)
    archived=torch.load(INITIAL,map_location='cpu',weights_only=False)
    model.base.load_state_dict(archived['model'])
    return model.to(device)


def data_to_gpu(arrays):
    return tuple(torch.as_tensor(a,device='cuda') for a in arrays[:3])+arrays[3:]


def aggregate(center,edge,valid,cohort,old,edge_absolute):
    def subset(mask):
        c=center[mask];v=valid[mask];passing=v&np.isfinite(c)&(c<.07);finite=v&np.isfinite(c)
        return dict(count=int(mask.sum()),passes=int(passing.sum()),recovery_rate=float(passing.mean()),
            invalid_count=int((~v).sum()),center_mean=float(c[finite].mean()) if finite.any() else None,
            center_median=float(np.median(c[finite])) if finite.any() else None,
            center_p90=float(np.quantile(c[finite],.9)) if finite.any() else None,
            edge_difference_mean=float(edge[mask][finite].mean()) if finite.any() else None,
            edge_absolute_pttd_mean=float(edge_absolute[mask][finite].mean()) if finite.any() else None)
    return dict(overall=subset(np.ones(len(center),bool)),groups={g:subset(cohort==i) for i,g in enumerate(GROUPS)},
                old_source=subset(old),new_source=subset(~old))


def evaluate(model,objective,arrays,batch_size,value_path=None):
    images,raw,targets,seeds,cohort,old=arrays;model.eval()
    values={k:[] for k in ('center','edge','valid','edge_absolute')}
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            sl=slice(start,start+batch_size)
            q=objective.corrected(model(images[sl],raw[sl]),targets[sl]);metrics=objective.optics.metrics(q)
            for key in ('center','edge','valid'):values[key].append(metrics[key].cpu().numpy())
            values['edge_absolute'].append(torch.sqrt(metrics['absolute'][:,1:].square().mean(-1)).amax(-1).cpu().numpy())
    values={k:np.concatenate(v) for k,v in values.items()}
    if value_path:np.savez_compressed(value_path,**values,seeds=seeds,cohort=cohort,old=old)
    return aggregate(values['center'],values['edge'],values['valid'],cohort,old,values['edge_absolute'])


def selection_key(metrics):
    o=metrics['overall']
    # All invalids are failures in the primary pass fraction; no edge score here.
    return (-o['passes'],o['center_p90'] if o['center_p90'] is not None else 1e9,
            o['center_mean'] if o['center_mean'] is not None else 1e9)


def save_candidate(model,epoch,metrics):
    path=HERE/'candidate.pt'
    save_torch(path,dict(state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,
        validation=metrics,architecture='CoordinatedNikon native147 + cross-module attention residual',
        base_source_sha256=sha256(INITIAL),raw_mean=model.raw_mean.cpu(),raw_std=model.raw_std.cpu(),
        target_mean=model.target_mean.cpu(),target_std=model.target_std.cpu(),
        input='same noisy raw3x49 and its original3x32x32 grid; sigma0.015waves',
        outputs='35 normalized Nikon coordinates grouped15/15/5; command=-clip(prediction,-1.5,1.5)'))
    atomic_json(HERE/'selected.json',dict(epoch=epoch,validation=metrics,path=str(path.resolve()),sha256=sha256(path),
        selection_rule='highest full6000 center WRMS<0.07 rate; tie lowest centerP90, then mean'))


def export_curves(history):
    atomic_json(HERE/'curves.json',dict(epochs=history,training_probe_count=1500,validation_count=6000,
        primary='single-shot center absolute PTTD WRMS<0.07waves',
        sensor_noise_std_waves=.015,experiments='one50-epoch candidate; no ablation or comparative training'))
    temp=HERE/'curves.partial'
    with temp.open('w',newline='',encoding='utf-8') as stream:
        w=csv.writer(stream);w.writerow(['epoch','train_loss','center_physical_loss','pose_loss','train_probe_recovery_rate',
            'validation_recovery_rate','G1_rate','G2_rate','G3_rate','old_validation_rate','new_validation_rate',
            'center_mean','center_p90','edge_difference_mean','edge_absolute_pttd_mean','invalid_validation','epoch_seconds'])
        for r in history:
            m=r['validation'];v=m['overall']
            w.writerow([r['epoch'],r['train_loss'],r.get('physical_loss'),r.get('pose_loss'),r['train_probe']['overall']['recovery_rate'],
                v['recovery_rate'],*[m['groups'][g]['recovery_rate'] for g in GROUPS],m['old_source']['recovery_rate'],
                m['new_source']['recovery_rate'],v['center_mean'],v['center_p90'],v['edge_difference_mean'],
                v['edge_absolute_pttd_mean'],v['invalid_count'],r['epoch_seconds']])
    os.replace(temp,HERE/'curves.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4));epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history],label='Center-led training objective')
    axes[0].set(xlabel='Epoch',ylabel='Loss')
    axes[1].plot(epochs,[100*r['train_probe']['overall']['recovery_rate'] for r in history],label='Fixed training probe1500')
    axes[1].plot(epochs,[100*r['validation']['overall']['recovery_rate'] for r in history],label='Validation6000')
    axes[1].axhline(90,ls=':',color='gray',label='90% target');axes[1].set(xlabel='Epoch',ylabel='One-shot center WRMS<0.07 (%)',ylim=(0,100))
    for g in GROUPS:axes[2].plot(epochs,[100*r['validation']['groups'][g]['recovery_rate'] for r in history],label=g)
    axes[2].set(xlabel='Epoch',ylabel='Validation recovery by source cohort (%)',ylim=(0,100))
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'curves.png',dpi=160);plt.close(fig)


def flight():
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    started=time.perf_counter();train,val,probe,fingerprints=load_data()
    raw_mean=train[1].mean(0);raw_std=np.maximum(train[1].std(0),.015)
    np.savez_compressed(HERE/'raw_normalization.npz',mean=raw_mean,std=raw_std)
    atomic_json(HERE/'data_manifest.json',dict(train_count=24000,validation_count=6000,samples_per_group=10000,
        new_sensor_observations=0,raw_input_available=True,noise_std_waves=.015,source_files=fingerprints,
        validation_seeds=val[3].tolist(),training_probe_seeds=probe[3].tolist(),
        raw_normalization='per-node mean/std from training only; no per-image normalization'))
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=fresh_model(raw_mean,raw_std);objective=CenterObjective(model)
    image=torch.tensor(train[0][:128],device='cuda');raw=torch.tensor(train[1][:128],device='cuda');target=torch.tensor(train[2][:128],device='cuda')
    model.eval()
    with torch.no_grad():
        difference=float((model(image,raw)-model.base(image)).abs().max())
    if difference!=0.:raise RuntimeError('Zero residual initialization did not preserve original predictor')
    reconstructed=nikon35.grid(train[1][:8])
    grid_difference=float(np.max(abs(reconstructed-train[0][:8])))
    if grid_difference>5e-5:raise RuntimeError('Raw nodes and image do not describe same sensor observation')
    # Corrected endpoints use the identical original32-interface nonlinear tracer.
    from validate import Reference
    reference=Reference(NikonTensorOptics())
    references=[]
    for index in (0,2000,4000):
        with torch.no_grad():
            x=torch.tensor(val[0][index:index+1],device='cuda');r=torch.tensor(val[1][index:index+1],device='cuda');y=torch.tensor(val[2][index:index+1],device='cuda')
            q=objective.corrected(model(x,r),y);actual=objective.optics.metrics(q)
        expected=reference.evaluate(q.cpu().numpy()[0]);error=abs(float(actual['center'][0])-expected['center'])
        if not bool(actual['valid'][0]) or error>1e-6:raise RuntimeError('Independent center endpoint check failed')
        references.append(dict(index=index,seed=int(val[3][index]),center_waves=expected['center'],error_waves=error))
    model.train();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True);loss,details=objective(model(image,raw),target);details['physical_loss'].backward()
    gradients={}
    for name,component in [('base_'+g,n) for g,n in zip(GROUPS,model.base.networks)]+[('coordination_'+g,h) for g,h in zip(GROUPS,model.correction_heads)]:
        norm=sum(float(p.grad.norm()) for p in component.parameters() if p.grad is not None)
        if norm<=0 or not math.isfinite(norm):raise RuntimeError(f'Missing center-loss gradient {name}')
        gradients[name]=norm
    # At zero output heads upstream gradients are exactly zero by design.
    # Check the raw and module pathways with a disposable nonzero head probe,
    # without an optimizer update or modifying the actual warm-start checkpoint.
    saved=[h.weight.detach().clone() for h in model.correction_heads]
    with torch.no_grad():
        for h in model.correction_heads:h.weight.normal_(0.,1e-3)
    model.eval();model.zero_grad(set_to_none=True)
    _,details=objective(model(image[:8],raw[:8]),target[:8]);details['physical_loss'].backward()
    for name,component in [('native_nodes',model.node_encoder),('native_dense',model.raw_global),('node_attention',model.node_attention),('module_attention',model.module_attention)]:
        norm=sum(float(p.grad.norm()) for p in component.parameters() if p.grad is not None)
        if norm<=0 or not math.isfinite(norm):raise RuntimeError(f'Missing coordination pathway {name}')
        gradients[name]=norm
    with torch.no_grad():
        for h,w in zip(model.correction_heads,saved):h.weight.copy_(w)
    # Autograd check for the NEW center objective along a correction direction.
    model.eval()
    with torch.no_grad():pred=model(image[:4],raw[:4])
    pred=pred.detach().requires_grad_(True)
    physical=objective(pred,target[:4])[1]['physical_loss']
    grad=torch.autograd.grad(physical,pred)[0]
    direction=torch.randn_like(pred);direction/=direction.norm()
    analytic=float((grad*direction).sum());step=1e-3
    with torch.no_grad():
        upper=objective(pred+step*direction,target[:4])[1]['physical_loss']
        lower=objective(pred-step*direction,target[:4])[1]['physical_loss']
    finite_difference=float((upper-lower)/(2*step));derivative_error=abs(analytic-finite_difference)/max(abs(analytic),abs(finite_difference),1e-8)
    if derivative_error>.005:raise RuntimeError(f'Center-loss derivative check failed {derivative_error}')
    model.train();timings=[]
    for _ in range(4):
        torch.cuda.synchronize();t=time.perf_counter();model.zero_grad(set_to_none=True)
        loss,details=objective(model(image,raw),target);loss.backward();torch.cuda.synchronize()
        timings.append(time.perf_counter()-t)
    batch_seconds=float(np.median(timings[1:]));parameters=sum(p.numel() for p in model.parameters());base_parameters=sum(p.numel() for p in model.base.parameters())
    train_seconds=50*math.ceil(24000/128)*batch_seconds
    report=dict(status='passed',gpu=torch.cuda.get_device_name(),batch_size=128,batch_seconds=batch_seconds,
        peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30,parameters=parameters,base_parameters=base_parameters,
        added_parameters=parameters-base_parameters,zero_residual_max_error=difference,raw_grid_max_error=grid_difference,
        physical_gradient_norms=gradients,independent_center_endpoints=references,
        directional_derivative=dict(autograd=analytic,finite_difference=finite_difference,relative_error=derivative_error),
        optimizer_updates=0,epochs=50,training_point_estimate_seconds=train_seconds,
        total_estimate_seconds=[train_seconds*1.05+300,train_seconds*1.5+900],
        source_hashes={p.name:sha256(p) for p in FILES},initial_checkpoint_sha256=sha256(INITIAL),
        optical_tracer_sha256=sha256(previous.PREVIOUS/'tensor_optics.py'),elapsed_seconds=time.perf_counter()-started)
    atomic_json(HERE/'flight.json',report);state('ready',flight=report)
    print('FLIGHT_PASSED',json.dumps(report),flush=True)


def independent_review(model,objective,val):
    from validate import Reference
    reference=Reference(NikonTensorOptics());rows=[]
    image,raw,target,seeds,cohort,old=val
    indices=[]
    for gi in range(3):
        for flag in (True,False):
            available=np.flatnonzero((cohort==gi)&(old==flag))
            indices.extend(available[np.linspace(0,len(available)-1,5,dtype=int)].tolist())
    model.eval()
    for index in indices:
        with torch.inference_mode():
            q=objective.corrected(model(image[index:index+1],raw[index:index+1]),target[index:index+1]);m=objective.optics.metrics(q)
        valid=bool(m['valid'][0]);row=dict(index=index,seed=int(seeds[index]),source_group=GROUPS[cohort[index]],valid=valid)
        if valid:
            ref=reference.evaluate(q.cpu().numpy()[0]);error=max(abs(float(m[k][0])-ref[k]) for k in ('center','edge'))
            if error>1e-6:raise RuntimeError(f'Independent terminal discrepancy {error}')
            row.update(center=ref['center'],edge_difference=ref['edge'],max_error_waves=error,passed=bool(ref['center']<.07))
        else:row['passed']=False
        rows.append(row);atomic_json(HERE/'independent_review30.json',dict(status='running',rows=rows,total=30))
    atomic_json(HERE/'independent_review30.json',dict(status='passed',rows=rows,total=30,scope='actual one-shot action, same noisy input, original RayOptics terminal check'))


def train():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    report=json.loads((HERE/'flight.json').read_text(encoding='utf-8'))
    if report['status']!='passed' or any(sha256(p)!=report['source_hashes'][p.name] for p in FILES):raise RuntimeError('Flight/source mismatch')
    if sha256(INITIAL)!=report['initial_checkpoint_sha256']:raise RuntimeError('Initialization checkpoint changed')
    if sha256(previous.PREVIOUS/'tensor_optics.py')!=report['optical_tracer_sha256']:raise RuntimeError('Validated optical tracer changed')
    started=time.perf_counter();train_np,val_np,probe_np,fingerprints=load_data()
    manifest=json.loads((HERE/'data_manifest.json').read_text(encoding='utf-8'))
    if fingerprints!=manifest['source_files']:raise RuntimeError('Input data changed')
    with np.load(HERE/'raw_normalization.npz') as n:raw_mean=n['mean'].copy();raw_std=n['std'].copy()
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    model=fresh_model(raw_mean,raw_std);objective=CenterObjective(model)
    # All existing encoders are trainable; new coordination learns faster.
    new_parameters=[p for name,p in model.named_parameters() if not name.startswith('base.')]
    optimizer=torch.optim.AdamW([dict(params=model.base.parameters(),lr=2e-5),dict(params=new_parameters,lr=1e-4)],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=50,eta_min=2e-6)
    train_data=data_to_gpu(train_np);val=data_to_gpu(val_np);probe=data_to_gpu(probe_np)
    plan=dict(version=HERE.name,epochs=50,batch_size=128,one_candidate=True,comparative_training=False,
        initialization='previous epoch29, highest center recovery; zero-initialized coordinated residual',
        initial_checkpoint=str(INITIAL.resolve()),initial_sha256=sha256(INITIAL),
        architecture='original three encoders +147 native noisy node branch + node attention + three-module self-attention +15/15/5 residual heads',
        trainable_parameters=report['parameters'],added_parameters=report['added_parameters'],
        primary_loss='mean[pseudoHuber(center/.07) +0.5*relu(center/.07-1)^2] +0.02 standardized pose SmoothL1(beta=.25)',
        edge_role='record absolute PTTD and bias-referenced difference; not training loss or model-selection score',
        action='paper L1: -clip(prediction,-1.5,1.5); labels only offline loss/scoring',
        samples_per_group=10000,train_count=24000,validation_count=6000,train_probe_count=1500,
        input_noise='same archived Gaussian0.015waves before grid interpolation; both branches same capture',
        raw_normalization='fixed training-only per-node statistics, not per-image amplitude normalization',
        optimizer=dict(base_lr=2e-5,coordination_lr=1e-4,weight_decay=1e-4,clip_norm=5.,cosine_epochs=50,min_lr=2e-6),
        selection='highest validation center WRMS<0.07 rate; tie lower centerP90 then mean; epoch0 retained',
        data_source_sha256=fingerprints,code_hashes=report['source_hashes'],trace_hash=report['optical_tracer_sha256'],
        GPU_data_cache=True,original_models_preserved=True)
    atomic_json(HERE/'plan.json',plan)
    backup=HERE/'initial_epoch029.pt'
    if not backup.exists():shutil.copyfile(INITIAL,backup)
    if sha256(backup)!=sha256(INITIAL):raise RuntimeError('Initial weight backup mismatch')
    resume=HERE/'resume.pt';history=[];epoch_start=1;batch_start=0;updates=0;accum=np.zeros(3);count=0;invalids=0;best=None;prior=0.;suspicious=[]
    if resume.exists():
        r=torch.load(resume,map_location='cuda',weights_only=False)
        if r['epochs']!=50 or r['source_hashes']!=report['source_hashes'] or r['initial_sha256']!=plan['initial_sha256']:raise RuntimeError('Resume protocol mismatch')
        model.load_state_dict(r['model']);optimizer.load_state_dict(r['optimizer']);scheduler.load_state_dict(r['scheduler'])
        history=r['history'];epoch_start=r['epoch'];batch_start=r['next_batch'];updates=r['updates'];accum=r['accum'];count=r['count'];invalids=r['invalids'];best=tuple(r['best']);prior=r['elapsed_seconds'];suspicious=r['timing_interruptions']
        torch.set_rng_state(r['cpu_rng'].cpu());torch.cuda.set_rng_state(r['cuda_rng'].cpu())
    else:
        state('initial_single_candidate_scoring')
        metrics=evaluate(model,objective,val,128,HERE/'epoch_000_validation.npz');probe_metrics=evaluate(model,objective,probe,128)
        history=[dict(epoch=0,train_loss=None,train_probe=probe_metrics,validation=metrics,epoch_seconds=0.)]
        best=selection_key(metrics);save_candidate(model,0,metrics);export_curves(history)
    def checkpoint(epoch,next_batch):
        save_torch(resume,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            history=history,epoch=epoch,next_batch=next_batch,updates=updates,accum=accum,count=count,invalids=invalids,best=best,
            epochs=50,source_hashes=report['source_hashes'],initial_sha256=plan['initial_sha256'],
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
            elapsed_seconds=prior+time.perf_counter()-started,timing_interruptions=suspicious))
    checkpoint(epoch_start,batch_start)
    batches=math.ceil(24000/128)
    for epoch in range(epoch_start,51):
        model.train();order=torch.tensor(np.random.RandomState(SEED+epoch).permutation(24000),device='cuda');t_epoch=time.perf_counter()
        for batch in range(batch_start,batches):
            t_batch=time.perf_counter();ix=order[batch*128:(batch+1)*128]
            image,raw,target=[a[ix] for a in train_data[:3]]
            optimizer.zero_grad(set_to_none=True);loss,details=objective(model(image,raw),target)
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
            accum+=np.array([float(loss),float(details['physical_loss']),float(details['pose_loss'])])*len(ix)
            count+=len(ix);invalids+=details['invalid_count'];updates+=1
            dt=time.perf_counter()-t_batch
            if dt>max(10.,10.*report['batch_seconds']):
                suspicious.append(dict(epoch=epoch,batch=batch+1,seconds=dt,time_unix=time.time(),kind='possible_suspend_or_external_stall; exclude_from_throughput_EST'))
            if (batch+1)%20==0 or batch+1==batches:
                checkpoint(epoch,batch+1)
                state('training',epoch=epoch,epochs=50,batch=batch+1,batches=batches,optimizer_updates=updates,
                    mean_loss=float(accum[0]/count),center_physical_loss=float(accum[1]/count),pose_loss=float(accum[2]/count),
                    invalid_training_endpoints=invalids,epoch_seconds=time.perf_counter()-t_epoch,
                    elapsed_seconds=prior+time.perf_counter()-started,timing_interruptions=suspicious)
                print('TRAIN',epoch,batch+1,batches,'loss',accum[0]/count,'invalid',invalids,flush=True)
        scheduler.step();state('epoch_curve_scoring',epoch=epoch,epochs=50,optimizer_updates=updates)
        metrics=evaluate(model,objective,val,128,HERE/f'epoch_{epoch:03d}_validation.npz')
        probe_metrics=evaluate(model,objective,probe,128)
        record=dict(epoch=epoch,train_loss=float(accum[0]/count),physical_loss=float(accum[1]/count),pose_loss=float(accum[2]/count),
            train_probe=probe_metrics,validation=metrics,invalid_training_endpoints=invalids,epoch_seconds=time.perf_counter()-t_epoch)
        history.append(record)
        save_torch(HERE/'epochs'/f'epoch_{epoch:03d}.pt',dict(model=model.state_dict(),epoch=epoch,validation=metrics))
        key=selection_key(metrics)
        if key<best:best=key;save_candidate(model,epoch,metrics)
        export_curves(history);print('EPOCH',epoch,json.dumps(metrics),flush=True)
        accum=np.zeros(3);count=0;invalids=0;batch_start=0;checkpoint(epoch+1,0)
    state('independent_review30',epochs=50,optimizer_updates=updates)
    selected=torch.load(HERE/'candidate.pt',map_location='cuda',weights_only=False);model.load_state_dict(selected['state_dict'])
    independent_review(model,objective,val)
    if sha256(INITIAL)!=plan['initial_sha256']:raise RuntimeError('Original epoch29 checkpoint changed')
    # Check the actual inference adapter on the identical saved noisy capture.
    from predict import Predictor
    predictor=Predictor(HERE/'candidate.pt',device='cuda',require_ready=False)
    with torch.inference_mode():expected=model.physical_prediction(val[0][:1],val[1][:1])[0].cpu().numpy()
    actual=predictor(val_np[1][0])
    interface_error=float(np.max(abs(actual-expected)))
    if interface_error>5e-5:raise RuntimeError(f'Prediction adapter mismatch {interface_error}')
    manifest=dict(state='ready_for_controller_use',checkpoint=str((HERE/'candidate.pt').resolve()),sha256=sha256(HERE/'candidate.pt'),
        selected_epoch=selected['epoch'],input='original noisy raw3x49; internally builds grid; no target/ideal input',
        output='normalized35 pose prediction; original paper L1 clipping in controller',inference_max_error=interface_error,
        automatic_validation='passed',assistant_review='pending',original_active_predictor_preserved=True)
    atomic_json(HERE/'candidate_manifest.json',manifest)
    final=dict(status='complete',epochs=50,optimizer_updates=updates,selected=json.loads((HERE/'selected.json').read_text(encoding='utf-8')),
        last_epoch=history[-1],elapsed_seconds=prior+time.perf_counter()-started,timing_interruptions=suspicious,
        background_running=False,assistant_review='pending',manifest=manifest)
    atomic_json(HERE/'summary.json',final);state('complete',**{k:v for k,v in final.items() if k!='status'})


def main(args):
    os.chdir(ROOT)
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        if args.flight:flight()
        elif (HERE/'summary.json').exists() and json.loads((HERE/'summary.json').read_text(encoding='utf-8'))['status']=='complete':print('ALREADY_COMPLETE',flush=True)
        else:train()
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--flight',action='store_true');args=parser.parse_args()
    try:main(args)
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise
