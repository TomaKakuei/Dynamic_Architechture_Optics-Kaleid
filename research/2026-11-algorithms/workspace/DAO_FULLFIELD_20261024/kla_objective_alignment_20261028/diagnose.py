"""Bounded KLA loss-landscape and train/eval normalization diagnosis.

Uses archived endpoints, not a controller: true endpoint gradients here are
training-objective diagnostics only. No result is a deployable recovery claim.
"""
from pathlib import Path
import sys, json, os, traceback
OUT=Path(__file__).resolve().parent
SOURCE=OUT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0,str(SOURCE))
from common import atomic, stamp, clock, awake, np
from optics import Optics, torch
from geometry import Geometry
torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
NAMES=['bounded_wfe','log_wfe','nominal_phase','psf_hellinger','mtf2000','otf_reference']
STEPS=[.0001,.0003,.001]

def state(stage,**kw):
    atomic(OUT/'state.json',dict(stage=stage,pid=os.getpid(),updated=stamp(),**kw))

def values(optics,q,reference=None):
    phase,mask=optics.residual(q); w=mask.double()
    rms=((phase.square()*w).sum(-1)/w.sum(-1).clamp_min(1)+1e-20).sqrt()
    ell=torch.log1p((rms/.07).square())
    pupil=torch.zeros((*phase.shape[:2],optics.grid,optics.grid),device=q.device,dtype=torch.complex128)
    pupil[:,:,optics.disk]=torch.exp(2j*torch.pi*phase)*w
    ft=torch.fft.fft2(pupil,s=(2*optics.grid,2*optics.grid))
    power=ft.abs().square(); psf=power/power.sum((-1,-2),keepdim=True).clamp_min(1e-30)
    otf=torch.fft.ifft2(power)
    otf=otf/otf[:,:,0:1,0:1].real.clamp_min(1e-30)
    n=int(optics.freq_index); f=optics.freq_index-n
    mtf=((1-f)*otf[:,:,n,0]+f*otf[:,:,n+1,0]).abs()
    score=mtf.mean(-1)-.3*((mtf-mtf.mean(-1,keepdim=True)).square().mean(-1)+1e-20).sqrt()
    result=dict(phase=phase,mask=mask,psf=psf,otf=otf,score=score,fields=mtf,wrms=rms,
                bounded_wfe=(ell/(1+ell)).mean(-1),log_wfe=ell.mean(-1),mtf2000=-score,
                valid=(w.mean(-1)>=.98).all(-1),support=w.mean(-1))
    if reference is not None:
        delta=phase-reference['phase']
        result['nominal_phase']=torch.log1p((delta.square()*w).sum(-1)/w.sum(-1).clamp_min(1)/.07**2).mean(-1)
        result['psf_hellinger']=((psf+1e-16).sqrt()-(reference['psf']+1e-16).sqrt()).square().sum((-1,-2)).mean(-1)
        result['otf_reference']=(otf-reference['otf']).abs().square().mean((-1,-2,-3))
        result['nominal_wrms']=((delta.square()*w).sum(-1)/w.sum(-1).clamp_min(1)+1e-20).sqrt()
    return result

def compact(v):
    return {k:v[k].detach().cpu().numpy() for k in ('score','fields','wrms','support','valid','nominal_wrms',*NAMES) if k in v}

def summarize(a):
    s=a['score']; valid=a['valid']
    return dict(count=len(s),passes=int((valid&(s>=.45)).sum()),invalid=int((~valid).sum()),
                q_mean=float(s.mean()),wrms_mean=float(a['wrms'].mean()),
                psf_hellinger_mean=float(a['psf_hellinger'].mean()) if 'psf_hellinger' in a else None)

def main():
    tick=clock();state('loading_saved_endpoints')
    with np.load(SOURCE/'training/joint_validation3000.npz') as z: saved={k:z[k].copy() for k in z.files}
    with np.load(SOURCE/'training/grid129_values.npz') as z: fine={k:z[k].copy() for k in z.files}
    rows=np.concatenate([np.where(saved['cohort']==c)[0][np.linspace(0,999,16,dtype=int)] for c in range(3)])
    protocol=dict(status='registered_before_compute',cases=48,rows=rows.tolist(),global_ids=saved['indices'][rows].tolist(),
        selection='16 equally spaced validation indices per cohort; not selected by results',grid=65,
        objectives=NAMES,max_normalized_coordinate_steps=STEPS,acquisitions=0,training_updates=0,
        inference_rerun='Only train/eval normalization diagnostics on these48; original baseline commands reused',
        gradient_scope='True-endpoint loss/gradient diagnosis for TRAINING objective design, not inference or deployed recovery',
        geometry='Original complete-path and >=98% pupil support',scoring='Original Q2000>=0.45 retained',
        psf_reference='Nominal fixed-image-plane PSF; retain defocus, remove PT exactly as original metric',
        bn_calibration='576 training images,192/cohort; no labels, no optimizer steps; fixed random mix')
    atomic(OUT/'protocol.json',protocol)
    w=saved['wrms']; qscore=fine['score']; wmax=w.max(1)
    atomic(OUT/'saved_distribution.json',dict(total=len(w),q129_passes=int((fine['valid']&(qscore>=.45)).sum()),
        wfe_all_fields_below007=int((wmax<.07).sum()),
        q_pass_wfe_fail=int(((qscore>=.45)&(wmax>=.07)).sum()),
        q_fail_wfe_pass=int(((qscore<.45)&(wmax<.07)).sum()),
        q_vs_mean_wfe_correlation=float(np.corrcoef(qscore,w.mean(1))[0,1]),
        mean_wrms_quantiles=np.quantile(w.mean(1),[0,.1,.5,.9,1]).tolist(),
        bounded_gradient_multiplier_quantiles=np.quantile(1/(1+np.log1p((w/.07)**2))**2,[0,.1,.5,.9,1]).tolist()))
    optic=Optics(grid=65,device='cuda'); geometry=Geometry()
    with torch.no_grad():ref=values(optic,torch.zeros((1,40),device='cuda',dtype=torch.float64))
    atomic(OUT/'nominal.json',{k:v.tolist() for k,v in compact(ref).items()})
    bases=[]; gradients=[]; candidates={f'{name}_{step:g}':[] for name in NAMES for step in STEPS}
    for start in range(0,len(rows),4):
        ix=rows[start:start+4];state('objective_gradients',completed=start,total=len(rows))
        x=torch.tensor(saved['endpoint'][ix],device='cuda',dtype=torch.float64,requires_grad=True)
        v=values(optic,x,ref); bases.append(compact(v)); gg=[]
        for j,name in enumerate(NAMES):
            g=torch.autograd.grad(v[name].sum(),x,retain_graph=j<len(NAMES)-1)[0].detach()
            gg.append(g.cpu().numpy())
        gradients.append(np.stack(gg,axis=1)); del v
        with torch.no_grad():
            for j,name in enumerate(NAMES):
                g=torch.as_tensor(gg[j],device='cuda'); d=-g/g.abs().amax(-1,keepdim=True).clamp_min(1e-30)
                for step in STEPS:
                    target=x.detach()+step*d
                    got=compact(values(optic,target,ref))
                    endpoint=target.cpu().numpy()
                    got['valid'] &= np.array([geometry.path(b,a) for a,b in zip(saved['initial'][ix],endpoint)])
                    candidates[f'{name}_{step:g}'].append(got)
        print(json.dumps(dict(stage='gradients',completed=start+len(ix),seconds=clock()-tick)),flush=True)
    cat=lambda chunks:{k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
    base=cat(bases); grad=np.concatenate(gradients); arrays=dict(rows=rows,ids=saved['indices'][rows],gradients=grad)
    arrays.update({f'base_{k}':v for k,v in base.items()})
    report=dict(baseline=summarize(base),gradient_alignment={},finite_steps={},active_seconds=clock()-tick,
                scope='Objective diagnostic on validation endpoints; no network/controller recovery improvement claimed')
    qgrad=grad[:,NAMES.index('mtf2000')]
    for j,name in enumerate(NAMES):
        g=grad[:,j];cos=(g*qgrad).sum(-1)/(np.linalg.norm(g,axis=-1)*np.linalg.norm(qgrad,axis=-1)).clip(1e-30)
        report['gradient_alignment'][name]=dict(mean_cosine_to_negative_Q=float(cos.mean()),
            median_cosine=float(np.median(cos)),opposes_Q_cases=int((cos<0).sum()),norm_median=float(np.median(np.linalg.norm(g,axis=-1))))
    for name,chunks in candidates.items():
        a=cat(chunks); arrays.update({f'{name}_{k}':v for k,v in a.items()})
        report['finite_steps'][name]=summarize(a)|dict(q_increased=int((a['score']>base['score']).sum()),
            q_decreased=int((a['score']<base['score']).sum()),
            psf_improved=int((a['psf_hellinger']<base['psf_hellinger']).sum()),
            gains=int((a['valid']&(a['score']>=.45)&(base['score']<.45)).sum()),
            losses=int(((~a['valid']|(a['score']<.45))&(base['score']>=.45)).sum()))
    np.savez_compressed(OUT/'landscape48.npz',**arrays)
    atomic(OUT/'objective_result.json',report)
    del optic,ref,x,grad;torch.cuda.empty_cache()
    # BatchNorm diagnosis: never claim per-batch statistics as one-shot inference.
    state('normalization_diagnostic',completed=0,total=2)
    from network import restore
    images=np.load(SOURCE/'cache/images.npy',mmap_mode='r')
    with np.load(SOURCE/'cache/metadata.npz') as z: meta={k:z[k].copy() for k in z.files}
    rng=np.random.default_rng(20261028)
    train_ids=np.concatenate([rng.choice(np.where((meta['cohort']==c)&(meta['indices']<9000))[0],192,replace=False) for c in range(3)])
    rng.shuffle(train_ids);np.save(OUT/'bn_training_ids.npy',train_ids)
    xs=saved['indices'][rows]; init=saved['initial'][rows]; optic=Optics(grid=65,device='cuda')
    @torch.no_grad()
    def commands(model):
        return np.concatenate([model.command(torch.tensor(images[ix].astype(np.float32),device='cuda')).cpu().numpy()
                               for ix in (xs[i:i+16] for i in range(0,len(xs),16))])
    @torch.no_grad()
    def evaluate(cmd):
        parts=[]
        for i in range(0,len(cmd),8):
            ep=init[i:i+8]+cmd[i:i+8];m=optic.metrics(torch.tensor(ep,device='cuda',dtype=torch.float64));a=compact(m)
            a['valid'] &= np.array([geometry.path(b,a0) for a0,b in zip(init[i:i+8],ep)])
            parts.append(a)
        return cat(parts)
    bnresult={}; bnarrays={}
    for filename in ('candidate.pt','last.pt'):
        model,payload=restore(SOURCE/'training'/filename)
        label=f"epoch{payload['epoch']}";bnresult[label]={}
        baseline=saved['command'][rows] if filename=='candidate.pt' else commands(model)
        bns=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
        bnresult[label]['batchnorm_layers']=len(bns)
        for stage in ('frozen','batch_statistics','training_recalibrated'):
            state('normalization_diagnostic',checkpoint=label,mode=stage)
            if stage=='frozen':cmd=baseline
            elif stage=='batch_statistics':
                for m in bns:m.train()
                cmd=commands(model)
            else:
                for m in bns:m.reset_running_stats();m.momentum=None;m.train()
                with torch.no_grad():
                    for i in range(0,len(train_ids),32):
                        im=torch.tensor(images[train_ids[i:i+32]].astype(np.float32),device='cuda');model.command(im)
                model.eval();cmd=commands(model)
            a=evaluate(cmd);bnresult[label][stage]=summarize(a)|dict(command_delta_rms=float(np.sqrt(np.mean((cmd-baseline)**2))))
            bnarrays[f'{label}_{stage}_command']=cmd
            for key,val in a.items():bnarrays[f'{label}_{stage}_{key}']=val
        del model;torch.cuda.empty_cache()
        atomic(OUT/'normalization_result.json',bnresult)
    np.savez_compressed(OUT/'normalization48.npz',**bnarrays)
    state('complete',cases=48,active_seconds=clock()-tick,training_updates=0,assistant_interpretation='pending')
    print(json.dumps(dict(stage='complete',seconds=clock()-tick,normalization=bnresult)),flush=True)

if __name__=='__main__':
    awake()
    try:main()
    except Exception as exc:
        state('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:awake(False)
