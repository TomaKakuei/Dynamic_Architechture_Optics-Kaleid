"""Posthoc diagnosis of22 fixed8step failures, never merged into the6000 score.
Selection uses evaluation failure only for diagnosis, not runtime routing.
"""
import json
import candidate as c
np,torch,run=c.np,c.torch,c.run
HERE=c.HERE


def main():
    if (HERE/'remaining22.json').exists():print('Already completed');return
    with np.load(HERE/'evaluation6000_4plus4/predictions.npz') as z:d={k:z[k].copy() for k in z.files}
    selected=np.flatnonzero(~d['valid']|(d['fields'].max(1)>=.07))
    model,system,optical=c.load();data=run.old.load_data(system)
    extra=np.load(run.HERE/'data/extra.npy',mmap_mode='r')
    lookup={int(v):i for i,v in enumerate(data['ids'])};rows=np.asarray([lookup[int(v)] for v in d['ids'][selected]])
    with torch.no_grad():
        image=torch.as_tensor(data['image'][rows],device='cuda');raw=torch.as_tensor(data['raw'][rows],device='cuda')
        ex=torch.as_tensor(np.array(extra[rows],copy=True),device='cuda');prior=c.prior(model,image,raw,ex)
        measured=torch.cat((raw.double().flatten(1)/7.,ex[:,:197].double()/np.sqrt(197),ex[:,197:].double()/7.),1)
        command8=torch.as_tensor(d['command'][selected],device='cuda')
        residual=optical.difference(-command8)-measured
        fieldfit=torch.cat((residual[:,:147].reshape(-1,3,49).square().mean(2).sqrt()*7.,
            residual[:,147:344].square().mean(1).sqrt()[:,None]*np.sqrt(197),
            residual[:,344:].reshape(-1,4,49).square().mean(2).sqrt()*7.),1)
        optical.trajectory_trace=[]
        command12=c.refine(command8,prior,raw,ex,optical)
        q=torch.as_tensor(d['initial'][selected],device='cuda')
        f,v,_=system.metrics(q+command12,q)
        report=dict(scope='Posthoc22failed cases only; not a6000recovery score or truth-based runtime rule',
            count=len(selected),further4_model_steps=4,new_acquisitions=0,training_updates=0,
            passes_after12_on_selected_failures=int((v&(f.max(1).values<.07)).sum()),
            cases=[dict(id=int(d['ids'][i]),active_bodies=int(d['active_bodies'][i]),
                fields8=d['fields'][i].tolist(),fields12=f[j].cpu().tolist(),
                observation_fit_worst_field=float(fieldfit[j].max())) for j,i in enumerate(selected)],
            steps=[dict(capped=int(t['cap_active'].sum()),accepted=int((t['gain']>0).sum())) for t in optical.trajectory_trace])
    (HERE/'remaining22.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez(HERE/'remaining22_values.npz',ids=d['ids'][selected],command12=command12.cpu().numpy(),fields12=f.cpu().numpy())
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
