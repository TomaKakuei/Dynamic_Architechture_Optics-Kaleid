"""Compare saved validation predictions only; no inference or model training."""
from pathlib import Path
import json,csv
from datetime import datetime,timezone
import numpy as np
HERE=Path(__file__).resolve().parent


def score(fields,valid,mask):
    f=fields[mask];good=valid[mask]&np.isfinite(f).all(1)
    passing=good&(f.max(1)<.07)
    return dict(count=int(mask.sum()),passes=int(passing.sum()),rate=float(passing.mean()),
        invalid=int((~good).sum()),field_mean=float(f[good].mean()) if good.any() else None)


def review(directory):
    history=json.loads((directory/'history.json').read_text())
    with np.load(directory/'baseline_validation.npz') as z:
        ids=z['original_ids'].copy();cohort=z['cohort'].copy();q=z['initial'].copy()
        initial_pass=z['valid']&np.isfinite(z['fields']).all(1)&(z['fields'].max(1)<.07)
    assert q.shape==(6000,35) and len(np.unique(ids))==6000
    active=np.any(q.reshape(-1,7,5)!=0,axis=2).sum(1)
    rows=[]
    for h in history:
        if h['epoch']>4:break
        path=directory/('baseline_validation.npz' if h['epoch']==0 else f"validation_{h['epoch']:03d}.npz")
        with np.load(path) as z:
            assert np.array_equal(z['original_ids'],ids) and np.array_equal(z['cohort'],cohort)
            assert np.array_equal(z['initial'],q)
            fields=z['fields'];valid=z['valid']
            assert fields.shape==(6000,3)
            all_rows=np.ones(6000,dtype=bool)
            overall=score(fields,valid,all_rows)
            assert overall['passes']==h['validation']['full_field_passes']
            passing=valid&np.isfinite(fields).all(1)&(fields.max(1)<.07)
            rows.append(dict(epoch=h['epoch'],training_loss=h.get('loss'),overall=overall,
                gained_from_initial=int((passing&~initial_pass).sum()),lost_from_initial=int((~passing&initial_pass).sum()),
                active_bodies={str(k):score(fields,valid,active==k) for k in np.unique(active)},
                cohorts={str(k):score(fields,valid,cohort==k) for k in np.unique(cohort)}))
    result=dict(status='reviewed_saved_predictions',scope='Original6000selection-validation cases; centerABS+twoedgeDIFF, all3<0.07waves',
        new_optical_evaluations=0,new_optimizer_updates=0,independent_test=False,
        original_ids_and_poses_identical=True,complete_four_epochs=bool(rows and rows[-1]['epoch']==4),
        rows=rows,updated=datetime.now(timezone.utc).isoformat())
    (directory/'saved_prediction_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    candidates={'old_large':HERE.parent/'original_routes_40ep_20261025/nikon',
                'fixed_bn':HERE/'fixed_bn_4ep','trajectory':HERE/'trajectory_4ep'}
    output={}
    for name,path in candidates.items():
        if (path/'history.json').exists():output[name]=review(path)
    (HERE/'four_epoch_comparison.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    with (HERE/'four_epoch_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['variant','epoch','passes','count','rate','all7_passes','all7_count','all7_rate','field_mean','gained','lost'])
        for name,result in output.items():
            for row in result['rows']:
                all7=row['active_bodies']['7'];overall=row['overall']
                writer.writerow([name,row['epoch'],overall['passes'],overall['count'],overall['rate'],all7['passes'],all7['count'],all7['rate'],overall['field_mean'],row['gained_from_initial'],row['lost_from_initial']])
    print(json.dumps({name:[dict(epoch=r['epoch'],passes=r['overall']['passes'],all7=r['active_bodies']['7']['passes'],gained=r['gained_from_initial'],lost=r['lost_from_initial']) for r in result['rows']] for name,result in output.items()}),flush=True)


if __name__=='__main__':main()
