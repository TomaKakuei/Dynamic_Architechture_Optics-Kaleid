"""Frozen paper cases, current single-acquisition controllers, no training."""
from pathlib import Path
import os, sys, json, time, argparse, traceback, ctypes, hashlib, csv
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(45, repeat=True)

ENV = Path(sys.prefix)
sys.path.append(str(ENV/'Lib/site-packages'))
_dll = os.add_dll_directory(str(ENV/'Library/bin'))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
HERE = Path(__file__).resolve().parent
sys.pycache_prefix = str(HERE/'local_pycache')
ROOT = HERE.parent
SOURCE = ROOT/'Adaptive Optics/ADAPTIVE_OPTICS_V15P'
WORK = ROOT/'DAO_FULLFIELD_20261024'
PAPER = SOURCE/'OL/DAO_Universal_v32_Restored/data'

def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.writing')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(30):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 29: raise
            time.sleep(.1)

def active():
    counter = ctypes.c_ulonglong()
    ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(counter))
    return counter.value / 1e7

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    published = read(PAPER/'published_results.json')
    seeds = {name: sorted({r['seed'] for r in published if r['system']==label})
             for name,label in [('trepan','Trepan2p'),('nikon','Nikon 100x'),('kla','KLA')]}
    assert all(len(v)==11 for v in seeds.values())
    hard = read(PAPER/'trepan_range_hard_cases.json')
    assert set(seeds['trepan']) == set(hard['selected_seeds'])
    trepan = {str(r['seed']):r['state'] for r in hard['selected']}
    kla = {}
    for seed in seeds['kla']:
        r = read(SOURCE/f'kaleid_modular_repair_20260906/results/kla/arch_a_{seed}.json')
        kla[str(seed)] = dict(state=r['initial_state'], definition=r['case_definition'])
    checkpoints = dict(trepan=str(WORK/'trepan_large/candidate.pt'),
        nikon=str(WORK/'nikon_prior_supervision_4ep_20261025/last.pt'),
        kla=str(WORK/'kla_objective_alignment_20261028/training_compare/old_wfe_stable/candidate.pt'))
    protocol = dict(seeds=seeds, count_per_system=11, checkpoints=checkpoints,
        checkpoint_sha256={k:sha(Path(v)) for k,v in checkpoints.items()},
        paper_registry=str(PAPER/'published_results.json'), paper_registry_sha256=sha(PAPER/'published_results.json'),
        paper_manuscript=str(ROOT/'DAO_FULL_MANUSCRIPT_20261029/DAO_manuscript_full.tex'),
        training_updates=0, observed_packets_per_case=1, physical_corrections_per_case=1,
        algorithms=dict(trepan='Frozen selected larger network epoch37, direct command',
                        nikon='Frozen prior-supervision E4, original4 native plus4 full540 model-only steps',
                        kla='Frozen old_wfe_stable E4, direct neural command, no feedback'),
        criteria=dict(trepan='All nine absolute PTTD WRMS <0.07 waves, finite, mechanical path and >=98% support',
                      nikon='Center absolute and +/-0.1mm edge differential PTTD WRMS all <0.07 waves, finite valid traces',
                      kla='Registered nine-field Q2000=mean(MTF)-0.3std(MTF)>=0.45 with valid trace/path; also report min(MTF)>=0.45 separately'),
        optics='Current frozen controller native optical models/calibration; exact archived physical perturbations. Not a replay of historical old optical implementation.',
        seed_policy='Fixed before inference: cameras 1933300000+case (KLA), 202610300000+case (Trepan); Nikon original RandomState(case) for native+center, registered SeedSequence for added outer channels.',
        invalid_policy='Keep all eleven cases; capture errors count as failures; no resampling or tuning',
        paper_and_source_unchanged=True)
    path = HERE/'protocol.json'
    if path.exists():
        assert read(path)==protocol, 'Protocol/source changed after freeze'
    else:
        save(path,protocol)
        save(HERE/'fixtures.json',dict(trepan=trepan,kla=kla))
    return protocol

def emit(system, row):
    save(HERE/system/'cases'/f"{row['seed']}.json",row)
    rows = [read(p) for p in (HERE/system/'cases').glob('*.json')]
    save(HERE/system/'state.json', dict(status='running' if len(rows)<11 else 'completed',
        pid=os.getpid(),completed=len(rows),total=11,passes=sum(r['pass'] for r in rows),
        effective_case_seconds=sum(r['active_seconds'] for r in rows)))
    print(system,row['seed'],row['pass'],row.get('fields',row.get('error')),flush=True)

def trepan(protocol):
    sys.path.insert(0,str(WORK))
    import experiment as exp
    np,torch=exp.np,exp.torch
    system=exp.System('trepan')
    model=system.model('larger')
    payload=torch.load(protocol['checkpoints']['trepan'],map_location='cpu',weights_only=False)
    assert payload['epoch']==37
    model.load_state_dict(payload['model'],strict=True);model.eval()
    from camera import Camera,WaveFit
    from common import GROUPS,SCALES
    cal=read(system.source/'camera_calibration.json')
    camera=Camera(calibration=cal);fit=WaveFit()
    fixtures=read(HERE/'fixtures.json')['trepan']
    for seed in protocol['seeds']['trepan']:
        if (HERE/'trepan/cases'/f'{seed}.json').exists():continue
        tick=active();state=fixtures[str(seed)]
        q=np.array([[state[str(a)][key] for key in ('dx_mm','dy_mm','dz_mm','tilt_x_deg','tilt_y_deg')]
                    for a,b in GROUPS])/SCALES
        q=q.reshape(40);seed_noise=202610300000+seed
        obsfile=HERE/'trepan/observations'/f'{seed}.npz';obsfile.parent.mkdir(parents=True,exist_ok=True)
        try:
            if obsfile.exists():
                with np.load(obsfile) as z:
                    assert np.array_equal(z['initial'],q);images=z['images'].copy()
            else:
                images=camera.capture(fit.fit(q),seed_noise,cal['grid'])
                np.savez_compressed(obsfile,initial=q,images=images)
            with torch.no_grad():
                image=torch.tensor(images[None],device='cuda',dtype=torch.float32)
                command=-torch.tanh(model(image)).double()
                initial=torch.tensor(q[None],device='cuda',dtype=torch.float64)
                fields,valid,_=system.metrics(initial+command,initial)
            f=fields[0].cpu().numpy();cmd=command[0].cpu().numpy()
            row=dict(seed=seed,initial=q.tolist(),command=cmd.tolist(),endpoint=(q+cmd).tolist(),
                fields=f.tolist(),valid=bool(valid[0]),pass_=bool(valid[0] and (f<.07).all()),
                center_pass=bool(valid[0] and f[0]<.07),mean_pass=bool(valid[0] and f.mean()<.07),
                all09_below08=bool(valid[0] and (f<.08).all()),noise_seed=seed_noise,
                active_seconds=active()-tick,observed_packets=1)
            row['pass']=row.pop('pass_')
        except (ValueError,RuntimeError) as exc:
            if 'out of memory' in str(exc).lower():raise
            row=dict(seed=seed,initial=q.tolist(),error=repr(exc),valid=False,**{'pass':False},
                     active_seconds=active()-tick,observed_packets=1)
        emit('trepan',row)

def nikon(protocol):
    sys.path.insert(0,str(WORK/'nikon_bottleneck_20261026'))
    import candidate as c
    np,torch=c.np,c.torch
    # The training factory unnecessarily reads a discarded warm-start file
    # from OneDrive before strict-loading the final frozen checkpoint. Supply
    # the same-shaped current base state at this construction-only stage;
    # every final parameter/buffer is still loaded and checked below.
    original_load=torch.load
    final_payload=original_load(c.CHECKPOINT,map_location='cpu',weights_only=False)
    final_state=final_payload['model']
    base_state={k:v for k,v in final_state.items() if not k.startswith('extra_')}
    skipped_path=Path(c.run.SOURCE).resolve()
    def construction_load(path,*args,**kwargs):
        if isinstance(path,(str,Path)) and Path(path).resolve()==skipped_path:
            return dict(model=base_state)
        return original_load(path,*args,**kwargs)
    torch.load=construction_load
    try:
        model,system,optical=c.load()
    finally:
        torch.load=original_load
    assert all(torch.equal(v.detach().cpu(),final_state[k]) for k,v in model.state_dict().items())
    save(HERE/'nikon/load_verification.json',dict(final_state_exact=True,
        checkpoint=str(c.CHECKPOINT),skipped_discarded_warmstart=str(skipped_path),
        reason='OneDrive warm-start recall stalls; final frozen checkpoint overwrites this initialization',
        numerical_algorithm_changed=False))
    n35=system.legacy.nikon35
    added=system.legacy.NikonTensorOptics(nodes=n35.NODES,fields=c.run.EXTRA_FIELDS,device='cuda')
    for seed in protocol['seeds']['nikon']:
        if (HERE/'nikon/cases'/f'{seed}.json').exists():continue
        tick=active();q,groups=n35.sample(seed)
        obsfile=HERE/'nikon/observations'/f'{seed}.npz';obsfile.parent.mkdir(parents=True,exist_ok=True)
        if obsfile.exists():
            with np.load(obsfile) as z:
                assert np.array_equal(z['initial'],q)
                raw=z['raw'].copy();extra=z['extra'].copy();maps=z['maps'].copy()
        else:
            # One original noisy phase capture at the exact initial state.
            plant=n35.Plant(q,noise_seed=seed,noise_std=.015,score_center=True)
            packet=plant.measure(np.zeros(35))
            if packet is None:raise RuntimeError(f'Original Nikon capture failed: {seed}')
            raw=np.asarray(packet['network'],dtype=np.float32)
            with torch.no_grad():
                target=torch.tensor(q[None],dtype=torch.float64,device='cuda')
                traced=added.trace(target+added.bias)
                signal=((traced['opl'][:,:,1:]-added.nominal_opl)@added.projector.T/added.wave)
                assert bool(traced['valid'].all())
                outer=signal[0].cpu().numpy()
            rng=np.random.default_rng(np.random.SeedSequence([2026102602,seed,496]))
            outer+=rng.normal(0.,.015,outer.shape)
            extra=np.r_[packet['center'],outer.ravel()].astype(np.float32)
            maps=n35.grid(raw)[0]
            np.savez_compressed(obsfile,initial=q,raw=raw,extra=extra,maps=maps)
        with torch.no_grad():
            tensors=[torch.tensor(a[None],dtype=torch.float32,device='cuda') for a in (maps,raw,extra)]
            command=c.predict(model,optical,*tensors)
            initial=torch.tensor(q[None],dtype=torch.float64,device='cuda')
            fields,valid,absolute=system.metrics(initial+command,initial)
        f=fields[0].cpu().numpy();a=absolute[0].cpu().numpy();cmd=command[0].cpu().numpy()
        row=dict(seed=seed,initial=q.tolist(),groups=groups,command=cmd.tolist(),endpoint=(q+cmd).tolist(),
            fields=f.tolist(),absolute_fields=a.tolist(),valid=bool(valid[0]),
            center_pass=bool(valid[0] and f[0]<.07),observed_packets=1,model_only_steps=8,
            active_seconds=active()-tick,**{'pass':bool(valid[0] and (f<.07).all())})
        emit('nikon',row)

def kla(protocol):
    sys.path.insert(0,str(SOURCE/'kla_joint_sensorless_20261008'))
    from common import np,GROUPS,SCALES
    from camera import Camera,WaveFit,torch
    from network import restore
    from optics import Optics
    from geometry import Geometry
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=SOURCE/'kla_joint_sensorless_20261008'
    model,_=restore(Path(protocol['checkpoints']['kla']));model.eval()
    camera=Camera(calibration=read(source/'camera_calibration.json'));fit=WaveFit();geo=Geometry()
    optics={g:Optics(grid=g,device='cuda') for g in (65,129,257)}
    fixtures=read(HERE/'fixtures.json')['kla']
    assert len(GROUPS)==8
    for seed in protocol['seeds']['kla']:
        if (HERE/'kla/cases'/f'{seed}.json').exists():continue
        tick=active();state=fixtures[str(seed)]['state']
        values=np.array([[state[f'L{i}'][key] for key in ('dx','dy','dz','tx','ty')] for i in range(6,14)])
        values[:,3:]=np.rad2deg(values[:,3:])
        q=(values/SCALES).ravel();seed_noise=1933300000+seed
        obsfile=HERE/'kla/observations'/f'{seed}.npz';obsfile.parent.mkdir(parents=True,exist_ok=True)
        try:
            if obsfile.exists():
                with np.load(obsfile) as z:
                    assert np.array_equal(z['initial'],q);images=z['images'].copy()
            else:
                images=camera.capture(fit.fit(q),seed_noise,2048)
                np.savez_compressed(obsfile,initial=q,images=images)
            with torch.no_grad():
                cmd=model.command(torch.tensor(images[None],dtype=torch.float32,device='cuda'))[0].double().cpu().numpy()
                endpoint=q+cmd
                target=torch.tensor(endpoint[None],dtype=torch.float64,device='cuda')
                low=optics[65].metrics(target);m=optics[129].metrics(target);grid=129
                if abs(float(m['score'][0])-.45)<.02 or (float(m['score'][0])>=.45)!=(float(low['score'][0])>=.45):
                    m=optics[257].metrics(target);grid=257
            f=m['fields'][0].cpu().numpy();score=float(m['score'][0])
            valid=bool(m['valid'][0] and low['valid'][0] and geo.path(endpoint,q))
            row=dict(seed=seed,initial=q.tolist(),command=cmd.tolist(),endpoint=endpoint.tolist(),fields=f.tolist(),
                Q2000=score,valid=valid,**{'pass':bool(valid and score>=.45)},
                all_fields_ge045=bool(valid and (f>=.45).all()),center_pass=bool(valid and f[4]>=.45),
                wrms=m['wrms'][0].cpu().numpy().tolist(),grid=grid,noise_seed=seed_noise,
                observed_packets=1,active_seconds=active()-tick)
        except (ValueError,RuntimeError) as exc:
            if 'out of memory' in str(exc).lower():raise
            row=dict(seed=seed,initial=q.tolist(),error=repr(exc),valid=False,**{'pass':False},
                     active_seconds=active()-tick,observed_packets=1)
        emit('kla',row)

def summarize(protocol):
    result={};flat=[]
    for name in ('trepan','nikon','kla'):
        rows=[read(HERE/name/'cases'/f'{s}.json') for s in protocol['seeds'][name]]
        assert len(rows)==11 and all(r['observed_packets']==1 for r in rows)
        assert sha(Path(protocol['checkpoints'][name]))==protocol['checkpoint_sha256'][name]
        result[name]=dict(passes=sum(r['pass'] for r in rows),total=11,
            passed_seeds=[r['seed'] for r in rows if r['pass']],failed_seeds=[r['seed'] for r in rows if not r['pass']],
            invalid_seeds=[r['seed'] for r in rows if not r['valid']],
            center_passes=sum(r.get('center_pass',False) for r in rows),
            effective_case_seconds=sum(r['active_seconds'] for r in rows))
        if name=='kla':result[name]['all_fields_ge045']=sum(r.get('all_fields_ge045',False) for r in rows)
        if name=='trepan':
            result[name]['mean_passes']=sum(r.get('mean_pass',False) for r in rows)
            result[name]['optical_only_all_fields_below07']=sum(bool(r.get('fields')) and max(r['fields'])<.07 for r in rows)
        for r in rows:
            f=r.get('fields',[])
            flat.append(dict(system=name,seed=r['seed'],valid=r['valid'],passed=r['pass'],
                criterion=protocol['criteria'][name],worst_field=(min(f) if name=='kla' else max(f)) if f else None,
                Q2000=r.get('Q2000'),fields=json.dumps(f),error=r.get('error','')))
    save(HERE/'result.json',dict(status='completed',systems=result,protocol=str(HERE/'protocol.json')))
    with (HERE/'per_case.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=flat[0].keys());writer.writeheader();writer.writerows(flat)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','trepan','nikon','kla','summary'])
    args=parser.parse_args()
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        print('SETUP',args.mode,'protocol',flush=True)
        protocol=prepare()
        print('SETUP',args.mode,'load_frozen_controller',flush=True)
        if args.mode=='summary':summarize(protocol)
        elif args.mode!='prepare':globals()[args.mode](protocol)
    except BaseException:
        save(HERE/f'{args.mode}_failure.json',dict(error=traceback.format_exc(),pid=os.getpid()))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
