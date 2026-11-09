"""Locate finite-difference branch changes without relaxing derivative tolerance."""
from check_trajectory_nikon import *


def main():
    system=old.make_system('nikon');data=old.load_data(system)
    original=json.loads((HERE/'trajectory_nikon_check.json').read_text())
    ix=np.array(original['training_indices'])
    torch.manual_seed(2026102501);model,_=old.make_model(system)
    model.load_state_dict(torch.load(old.HERE/'nikon/candidate.pt',map_location='cpu',weights_only=False)['model'])
    model.__class__=FixedStatsNetwork;model.eval()
    image=torch.tensor(data['image'][ix],device='cuda');raw=torch.tensor(data['raw'][ix],device='cuda')
    target=torch.tensor(data['q'][ix],device='cuda',dtype=torch.float64)
    with torch.no_grad():
        neural=exp.LargerNetwork.forward(model,image,raw)
        prior=(neural*model.base.target_std+model.base.target_mean).clamp(-1.5,1.5).double()
    measured=raw.flatten(1).double()/7.
    def objective(physical):
        fields,valid,_=system.metrics(target-physical,target);assert bool(valid.all())
        ell=torch.log1p((fields/.07).square());term=ell/(1+ell)
        control=.5*term.mean(1)+.5*term.amax(1)
        label=(target-model.base.target_mean.double())/model.base.target_std.double()
        estimate=(physical-model.base.target_mean.double())/model.base.target_std.double()
        pose=torch.nn.functional.smooth_l1_loss(estimate,label,beta=.25,reduction='none').mean(1)
        return control+.02*pose,fields.argmax(1).cpu().numpy()
    model.trajectory_trace=[]
    variable=prior.clone().requires_grad_(True)
    terminal=four_steps(variable,measured,model)
    loss,base_worst=objective(terminal)
    gradient=torch.autograd.grad(loss.mean(),variable)[0]
    baseline_trace=model.trajectory_trace
    generator=torch.Generator(device='cuda').manual_seed(2026102522)
    directions=dict(exact_gradient=gradient,random=torch.randn(prior.shape,dtype=prior.dtype,device='cuda',generator=generator))
    report=dict(training_indices=ix.tolist(),validation_cases=0,optimizer_updates=0,
        original_registered_failure_preserved=True,relative_tolerance=.002,
        reason='Test whether original steps crossed discrete solver or worst-field branches; tolerance unchanged',
        per_sample_gradient_norm=gradient.norm(dim=1).cpu().tolist(),rows=[])
    for name,direction in directions.items():
        direction=direction/direction.square().mean().sqrt()
        predicted=float((gradient*direction).sum())
        for h in (1e-4,3e-5,1e-5,3e-6,1e-6):
            values=[];traces=[];worsts=[]
            for sign in (-1,1):
                model.trajectory_trace=[]
                with torch.no_grad():
                    result=four_steps(prior+sign*h*direction,measured,model)
                    value,worst=objective(result)
                values.append(value.cpu().numpy());traces.append(model.trajectory_trace);worsts.append(worst)
            derivative=float((values[1].mean()-values[0].mean())/(2*h))
            changes=[]
            for sign,trace,worst in zip((-1,1),traces,worsts):
                altered=[]
                for step,(base,now) in enumerate(zip(baseline_trace,trace)):
                    for key in ('gain','cap_index','cap_active','clamp_mask'):
                        rows=np.any(base[key]!=now[key],axis=1)
                        if rows.any():altered.append(dict(step=step,key=key,rows=np.flatnonzero(rows).tolist()))
                if np.any(worst!=base_worst):altered.append(dict(key='worst_field',rows=np.flatnonzero(worst!=base_worst).tolist()))
                changes.append(dict(sign=sign,changes=altered))
            report['rows'].append(dict(direction=name,h=h,predicted=predicted,observed=derivative,
                relative_error=abs(predicted-derivative)/max(abs(predicted),abs(derivative),1e-8),
                changes=changes,per_case_observed_derivative=((values[1]-values[0])/(2*h)).tolist()))
            atomic(HERE/'branch_window_check.json',report)
            print(json.dumps({k:v for k,v in report['rows'][-1].items() if k!='per_case_observed_derivative'}),flush=True)
    del model.trajectory_trace


if __name__=='__main__':main()
