"""Frozen prior, four original GN steps, add the existing measured channels.

No new acquisition or training. Same 200 selected seven-body cases; compare
to saved native-only results. This is a diagnostic candidate, not deployment.
"""
from pathlib import Path
import json,sys
import diagnose as diag
run,np,torch=diag.run,diag.np,diag.torch
HERE=Path(__file__).resolve().parent


class AllSensors:
    def __init__(self,system):
        legacy=system.legacy
        self.diff=run.make_typed_optics(legacy.NikonTensorOptics(
            nodes=legacy.nikon35.NODES,fields=(0.,-.05,.05,-.1,.1,-.15,.15),device='cuda'),torch.float64)
        self.center=run.make_typed_optics(legacy.NikonTensorOptics(
            nodes=legacy.nikon35.AUDIT_NODES,fields=(0.,),device='cuda'),torch.float64)
        self.trajectory_trace=None

    def difference(self,physical):
        t=self.diff.trace(physical.double()+self.diff.bias)
        d=(t['opl'][:,:,1:]-self.diff.nominal_opl)@self.diff.projector.T/self.diff.wave
        c=diag.center_wave(self.center,physical)
        return torch.cat((d[:,:3].flatten(1)/7.,c/(197**.5),d[:,3:].flatten(1)/7.),1)

    @torch.no_grad()
    def response(self,physical):
        def one(q):return self.difference(q[None])[0]
        with torch.inference_mode(False),torch.no_grad():
            return torch.func.vmap(torch.func.jacfwd(one))(physical.clone())


class NativeSensors:
    def __init__(self,system):
        legacy=system.legacy
        self.native=run.make_typed_optics(legacy.NikonTensorOptics(
            nodes=legacy.nikon35.NODES,fields=(0.,-.05,.05),device='cuda'),torch.float64)
        self.trajectory_trace=None

    def difference(self,physical):
        t=self.native.trace(physical.double()+self.native.bias)
        return (((t['opl'][:,:,1:]-self.native.nominal_opl)@self.native.projector.T/self.native.wave)/7.).flatten(1)

    @torch.no_grad()
    def response(self,physical):
        from fp32_response import response
        return response(self.native,physical)


def main():
    staged='--staged' in sys.argv
    label='sensor_staged_control' if staged else 'sensor_control'
    result=HERE/(label+'.json')
    if result.exists():print(result.read_text());return
    tick=run.exp.active_time()
    run.exp.atomic(HERE/(label+'_state.json'),dict(status='loading',updates=0))
    with np.load(HERE/'values.npz') as z:baseline={k:z[k].copy() for k in z.files}
    system=run.old.make_system('nikon')
    data=run.old.load_data(system)
    lookup={int(identity):i for i,identity in enumerate(data['ids'])}
    rows=np.asarray([lookup[int(identity)] for identity in baseline['ids']])
    extra=np.load(run.HERE/'data/extra.npy',mmap_mode='r')
    saved=torch.load(run.HERE/'last.pt',map_location='cpu',weights_only=False)['model']
    mean=saved['base.target_mean'].cuda();std=saved['base.target_std'].cuda()
    optical=AllSensors(system)
    native=NativeSensors(system) if staged else None
    fields,valid,commands,fit,trace=[],[],[],[],[]
    with torch.no_grad():
        for start in range(0,len(rows),50):
            end=min(start+50,len(rows));selected=rows[start:end]
            raw=torch.as_tensor(data['raw'][selected],device='cuda').double().flatten(1)
            ex=torch.as_tensor(np.array(extra[selected],copy=True),device='cuda').double()
            measured=torch.cat((raw/7.,ex[:,:197]/(197**.5),ex[:,197:]/7.),1)
            prior=torch.as_tensor(baseline['prior'][start:end],device='cuda')
            optical.trajectory_trace=[]
            if staged:
                from trajectory_backward import TrajectoryStep
                physical=prior
                for _ in range(3):physical=TrajectoryStep.apply(physical,prior,measured[:,:147],native)
                physical=TrajectoryStep.apply(physical,prior,measured,optical)
            else:
                physical=run.four_steps(prior,measured,optical)
            trace.append(optical.trajectory_trace);optical.trajectory_trace=None
            normalized=((physical-mean)/std).float()
            command=-(normalized*std+mean).clamp(-1.5,1.5)
            target=torch.as_tensor(data['q'][selected],device='cuda').double()
            f,v,_=system.metrics(target+command.double(),target)
            residual=optical.difference(-command.double())-measured
            fit.append(torch.stack((residual[:,:147].square().mean(1).sqrt()*7.,
                residual[:,147:344].square().mean(1).sqrt()*(197**.5),
                residual[:,344:].square().mean(1).sqrt()*7.),1).cpu().numpy())
            commands.append(command.cpu().numpy());fields.append(f.cpu().numpy());valid.append(v.cpu().numpy())
            run.exp.atomic(HERE/(label+'_state.json'),dict(status='running',completed=end,total=len(rows)))
    fields=np.concatenate(fields);valid=np.concatenate(valid);commands=np.concatenate(commands);fit=np.concatenate(fit)
    report=dict(scope=dict(cases=200,training_updates=0,neural_prior='reuse frozen additive E4 prior',
        acquisitions=1,steps=4,observation_count=540,edge_definition='DIFF',center_added='ABS',
        sensor_scaling='each field divided by square root of its node count; original147 weighting unchanged',
        ridge=1e-4,cap=.25,gains=[1.,.5,.25],truth_used_only_for_scoring=True,
        diagnostic_candidate_only=True,independent_review=False),groups={})
    report['scope']['schedule']='3native + 1all' if staged else '4all'
    for name,sl in [('training',slice(0,100)),('validation',slice(100,200))]:
        oldpass=baseline['valid'][sl]&(baseline['fields'][sl].max(1)<.07)
        newpass=valid[sl]&(fields[sl].max(1)<.07)
        report['groups'][name]=dict(count=100,original_passes=int(oldpass.sum()),passes=int(newpass.sum()),
            gained=int((~oldpass&newpass).sum()),lost=int((oldpass&~newpass).sum()),
            center_passes=int((valid[sl]&(fields[sl,0]<.07)).sum()),invalid=int((~valid[sl]).sum()),
            final_fields_mean=fields[sl].mean(0).tolist(),
            pose_rms=diag.describe(np.sqrt(np.mean((commands[sl]+baseline['original_q'][sl])**2,1))),
            sensor_fit={key:diag.describe(fit[sl,i]) for i,key in enumerate(['native','center_abs','outer_diff'])})
    report['active_seconds']=run.exp.active_time()-tick
    report['updated']=run.exp.stamp()
    np.savez(HERE/(label+'_values.npz'),ids=baseline['ids'],fields=fields,valid=valid,command=commands,fit=fit)
    run.exp.atomic(result,report)
    run.exp.atomic(HERE/(label+'_state.json'),dict(status='completed',active_seconds=report['active_seconds']))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
