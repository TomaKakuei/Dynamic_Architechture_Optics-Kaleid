"""User-requested original6000 evaluation, reusing the completed final100.

No training, repeated prediction, independent review or alternate criterion.
"""
from pathlib import Path
import json
import os
import traceback
import numpy as np
import torch
import run_trial as run

OUT = run.HERE/'evaluation6000'


def progress(status, **kw):
    run.exp.atomic(OUT/'state.json',dict(status=status,pid=os.getpid(),updated=run.exp.stamp(),**kw))


def main():
    import msvcrt
    OUT.mkdir(exist_ok=True)
    lock = (OUT/'evaluation.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    if (OUT/'result.json').exists():
        print((OUT/'result.json').read_text());return
    if os.name=='nt':
        run.ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        progress('loading',training_updates=0)
        system=run.old.make_system('nikon');system.__class__=run.AdditiveSystem
        data=run.old.load_data(system)
        data['extra']=np.load(run.HERE/'data/extra.npy',mmap_mode='r')
        indices=np.flatnonzero(~data['train'])
        if len(indices)!=6000:
            raise RuntimeError('Expected original6000 validation rows')
        checkpoint=torch.load(run.HERE/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['epoch']!=4:
            raise RuntimeError('Expected completed epoch4 checkpoint')
        stats=dict(mean=checkpoint['model']['extra_mean'].numpy(),std=checkpoint['model']['extra_std'].numpy())
        model=run.make_model(system,stats)
        model.load_state_dict(checkpoint['model'],strict=True);model.eval();del checkpoint
        with np.load(run.HERE/'final_sample100.npz') as z:
            cached={k:z[k].copy() for k in z.files}
        slots={int(identity):row for row,identity in enumerate(data['ids'][indices])}
        cached_slots=np.asarray([slots[int(identity)] for identity in cached['original_ids']])
        if len(cached_slots)!=100 or len(np.unique(cached_slots))!=100:
            raise RuntimeError('Expected100 cached final predictions with original IDs')
        w=np.empty((6000,3),dtype=cached['fields'].dtype)
        absolute=np.empty((6000,3),dtype=cached['absolute'].dtype)
        command=np.empty((6000,35),dtype=cached['command'].dtype)
        valid=np.zeros(6000,dtype=bool);completed=np.zeros(6000,dtype=bool)
        for target,key in ((w,'fields'),(absolute,'absolute'),(command,'command'),(valid,'valid')):
            target[cached_slots]=cached[key]
        completed[cached_slots]=True
        missing_slots=np.flatnonzero(~completed)
        run.exp.atomic(OUT/'protocol.json',dict(count=6000,original_ids=data['ids'][indices].tolist(),
            checkpoint='last.pt',epoch=4,training_updates=0,
            criterion='Center ABS + both +/-0.1mm DIFF, each <0.07waves, valid traces',
            fields_mm=[0.,-.1,.1],audit_pupil_nodes=197,
            measured_acquisitions=1,model_only_refinements=4,measured_feedback=False,
            reused_final_predictions=100,new_predictions_needed=5900,independent_review=False))
        tick=run.exp.active_time();new_count=0
        with torch.no_grad():
            for first in range(0,len(missing_slots),run.BATCH):
                block_slots=missing_slots[first:first+run.BATCH]
                ix=indices[block_slots]
                path=OUT/f'part_{first//run.BATCH:03d}.npz'
                if path.exists():
                    with np.load(path) as z:
                        if not np.array_equal(z['original_ids'],data['ids'][ix]):
                            raise RuntimeError('Cached evaluation chunk ID mismatch')
                        block={k:z[k].copy() for k in ('fields','valid','absolute','command')}
                else:
                    endpoint,initial,cmd,_=system.forward(model,data,ix)
                    fields,good,abs_fields=system.metrics(endpoint,initial)
                    block=dict(fields=fields.cpu().numpy(),valid=good.cpu().numpy(),
                               absolute=abs_fields.cpu().numpy(),command=cmd.cpu().numpy())
                    np.savez(path,original_ids=data['ids'][ix],**block)
                    new_count+=len(ix)
                for target,key in ((w,'fields'),(absolute,'absolute'),(command,'command'),(valid,'valid')):
                    target[block_slots]=block[key]
                completed[block_slots]=True
                seconds=run.exp.active_time()-tick
                progress('evaluating',completed=int(completed.sum()),total=6000,
                    reused_final100=100,new_predictions_this_invocation=new_count,
                    active_seconds=seconds,
                    remaining_seconds=seconds/max(1,new_count)*int((~completed).sum()))
        result=run.exp.aggregate(w,valid,absolute,.07)
        result['cohorts']={str(group):run.exp.aggregate(w[data['cohort'][indices]==group],
            valid[data['cohort'][indices]==group],absolute[data['cohort'][indices]==group],.07)
            for group in np.unique(data['cohort'][indices])}
        np.savez(OUT/'final_validation6000.npz',indices=indices,original_ids=data['ids'][indices],
            cohort=data['cohort'][indices],initial=data['q'][indices],command=command,
            fields=w,valid=valid,absolute=absolute)
        # Historical reference is read from already-saved outcomes, never rerun.
        reference=run.ROOT/'nikon_large_debug_20261025/trajectory_4ep/validation_003.npz'
        with np.load(reference) as z:
            lookup={int(identity):row for row,identity in enumerate(z['original_ids'])}
            order=np.asarray([lookup[int(identity)] for identity in data['ids'][indices]])
            old_fields=z['fields'][order];old_valid=z['valid'][order]
        old_pass=old_valid & np.isfinite(old_fields).all(1) & (old_fields.max(1)<.07)
        new_pass=valid & np.isfinite(w).all(1) & (w.max(1)<.07)
        report=dict(status='completed',epoch=4,training_updates=0,validation=result,
            historical_E3=dict(passes=int(old_pass.sum()),count=6000,rate=float(old_pass.mean()),
                center_passes=int((old_valid&(old_fields[:,0]<.07)).sum()),
                field_mean=float(old_fields[old_valid].mean()),source=str(reference),new_inference=False),
            change=dict(passes=int(new_pass.sum()-old_pass.sum()),
                percentage_points=float(100*(new_pass.mean()-old_pass.mean())),
                gained=int((new_pass&~old_pass).sum()),lost=int((~new_pass&old_pass).sum())),
            cached_final100_reused=True,new_predictions=5900,independent_review=False,
            active_seconds=run.exp.active_time()-tick,updated=run.exp.stamp())
        run.exp.atomic(OUT/'result.json',report)
        progress('completed',completed=6000,total=6000,passes=result['full_field_passes'],
                 rate=result['full_field_rate'],training_updates=0)
        print(json.dumps(report,indent=2),flush=True)
    except BaseException:
        progress('failed',error=traceback.format_exc());raise
    finally:
        if os.name=='nt':
            run.ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':
    main()
