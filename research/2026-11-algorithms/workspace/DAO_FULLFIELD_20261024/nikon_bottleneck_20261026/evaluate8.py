"""Original6000 evaluation of the fixed4+4 candidate; reuse48 completed endpoints.
Reuse saved native4 outputs for every case. Zero training, no extra acquisition.
"""
from pathlib import Path
import json,os,traceback
import candidate as c
np,torch,run=c.np,c.torch,c.run
HERE=c.HERE;OUT=HERE/'evaluation6000_4plus4'


def state(status,**kw):run.exp.atomic(OUT/'state.json',dict(status=status,pid=os.getpid(),**kw))


def aggregate(f,v):
    good=v&np.isfinite(f).all(1);p=good&(f.max(1)<.07)
    return dict(count=len(f),passes=int(p.sum()),rate=float(p.mean()),invalid=int((~good).sum()),
        center_passes=int((good&(f[:,0]<.07)).sum()),per_field_mean=f[good].mean(0).tolist(),
        field_mean=float(f[good].mean()),worst_p90=float(np.quantile(f[good].max(1),.9)))


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'result.json').exists():print((OUT/'result.json').read_text());return
    state('loading');model,system,optical=c.load();data=run.old.load_data(system)
    added=np.load(run.HERE/'data/extra.npy',mmap_mode='r')
    baseline_path=c.ROOT/'nikon_prior_supervision_4ep_20261025/evaluation6000/final_validation6000.npz'
    with np.load(baseline_path) as z:old={k:z[k].copy() for k in z.files}
    ids=old['original_ids'];lookup={int(v):i for i,v in enumerate(data['ids'])}
    rows=np.asarray([lookup[int(v)] for v in ids])
    assert len(rows)==6000 and not data['train'][rows].any()
    command=np.empty((6000,35));fields=np.empty((6000,3));valid=np.zeros(6000,bool);done=np.zeros(6000,bool)
    with np.load(HERE/'values.npz') as z:
        slotlookup={int(v):i for i,v in enumerate(ids)}
        slots=np.asarray([slotlookup[int(v)] for v in z['ids']])
        command[slots]=-z['all540_noisy_8_estimate'];fields[slots]=z['all540_noisy_8_fields'];valid[slots]=True;done[slots]=True
    missing=np.flatnonzero(~done)
    run.exp.atomic(OUT/'protocol.json',dict(count=6000,case_ids=ids.tolist(),
        checkpoint=str(c.CHECKPOINT),training_updates=0,acquisitions=1,
        native_steps=4,all540_steps=4,total_model_steps=8,truth_used_only_for_scoring=True,
        fields_mm=[0,-.1,.1],nodes=197,criterion='CenterABS and two edgeDIFF all<0.07waves; valid',
        baseline_native4_outputs_reused=6000,candidate8_endpoints_reused=48,new_extensions=5952,
        preliminary48_in_validation=True,independent_test=False,independent_review=False,automatic_deployment=False))
    tick=run.exp.active_time();new_count=0
    with torch.no_grad():
        for start in range(0,len(missing),128):
            block_slots=missing[start:start+128];ix=rows[block_slots];path=OUT/f'part_{start//128:03d}.npz'
            if path.exists():
                with np.load(path) as z:
                    if not np.array_equal(z['ids'],ids[block_slots]):raise RuntimeError('Cached IDs differ')
                    block={k:z[k].copy() for k in ('command','fields','valid')}
            else:
                image=torch.as_tensor(data['image'][ix],device='cuda')
                raw=torch.as_tensor(data['raw'][ix],device='cuda')
                extra=torch.as_tensor(np.array(added[ix],copy=True),device='cuda')
                initial_prior=c.prior(model,image,raw,extra)
                command4=torch.as_tensor(old['command'][block_slots],device='cuda')
                cmd=c.refine(command4,initial_prior,raw,extra,optical)
                target=torch.as_tensor(old['initial'][block_slots],device='cuda').double()
                f,v,_=system.metrics(target+cmd,target)
                block=dict(command=cmd.cpu().numpy(),fields=f.cpu().numpy(),valid=v.cpu().numpy())
                np.savez(path,ids=ids[block_slots],**block);new_count+=len(ix)
            command[block_slots]=block['command'];fields[block_slots]=block['fields'];valid[block_slots]=block['valid'];done[block_slots]=True
            elapsed=run.exp.active_time()-tick
            state('running',completed=int(done.sum()),total=6000,active_seconds=elapsed,
                  remaining_seconds=elapsed/max(1,new_count)*int((~done).sum()))
    p=valid&np.isfinite(fields).all(1)&(fields.max(1)<.07)
    op=old['valid']&np.isfinite(old['fields']).all(1)&(old['fields'].max(1)<.07)
    active=(np.abs(old['initial'].reshape(-1,7,5)).max(2)>1e-10).sum(1)
    report=dict(status='completed',candidate=aggregate(fields,valid),baseline=aggregate(old['fields'],old['valid']),
        change=dict(gained=int((p&~op).sum()),lost=int((~p&op).sum()),net=int(p.sum()-op.sum()),
                    percentage_points=float(100*(p.mean()-op.mean()))),
        active_bodies={str(n):dict(candidate=aggregate(fields[active==n],valid[active==n]),
            baseline=aggregate(old['fields'][active==n],old['valid'][active==n])) for n in np.unique(active)},
        acquisitions=1,model_steps=8,training_updates=0,independent_review=False,automatic_deployment=False,
        active_seconds=run.exp.active_time()-tick,updated=run.exp.stamp())
    np.savez(OUT/'predictions.npz',ids=ids,initial=old['initial'],command=command,fields=fields,valid=valid,active_bodies=active)
    run.exp.atomic(OUT/'result.json',report);state('completed',passes=int(p.sum()),total=6000,active_seconds=report['active_seconds'])
    print(json.dumps({k:v for k,v in report.items() if k!='active_bodies'},indent=2))


if __name__=='__main__':
    try:main()
    except BaseException:state('failed',error=traceback.format_exc());raise
