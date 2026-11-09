"""CPU-only coverage analysis of existing Nikon samples and saved predictions.

No optical trace, recovery-network inference/training, or numerical replay.
Small domain classifiers are statistical diagnostics, not recovery models.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import json,time,csv,traceback
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp,spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
TRIAL=ROOT/'nikon_additive_4ep_20261025'
SEED=2026102607


def save(name,value):
    path=HERE/name
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def state(stage,**kw):
    save('state.json',dict(stage=stage,pid=os.getpid(),**kw))
    print(json.dumps(dict(stage=stage,**kw)),flush=True)


def describe(x):
    x=np.asarray(x)
    return dict(count=len(x),mean=float(np.mean(x)),std=float(np.std(x)),
        p05=float(np.quantile(x,.05)),median=float(np.median(x)),
        p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)))


def compare(a,b):
    scale=max(float(np.sqrt((np.var(a)+np.var(b))/2)),1e-12)
    return dict(a=describe(a),b=describe(b),standardized_mean_difference=float((np.mean(b)-np.mean(a))/scale),
                ks_statistic=float(ks_2samp(a,b).statistic))


def rate(mask,passed):
    n=int(np.sum(mask));k=int(np.sum(passed[mask]))
    return dict(count=n,passes=k,rate=k/n if n else None)


def binned(values,passed,old,edges):
    rows=[]
    for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
        mask=(values>=lo)&((values<=hi) if i==len(edges)-2 else (values<hi))
        rows.append(dict(bin=i+1,lower=None if not np.isfinite(lo) else float(lo),
            upper=None if not np.isfinite(hi) else float(hi),**rate(mask,passed),
            old_passes=int(old[mask].sum()),value_mean=float(values[mask].mean()) if mask.any() else None))
    return rows


def domain_auc(features,train_rows,val_rows):
    """One fixed lightweight train/validation-domain classifier, held-out AUC."""
    rng=np.random.default_rng(SEED)
    selected=rng.choice(train_rows,len(val_rows),replace=False)
    x=np.r_[features[selected],features[val_rows]]
    y=np.r_[np.zeros(len(selected),dtype=int),np.ones(len(val_rows),dtype=int)]
    fit,test=train_test_split(np.arange(len(y)),test_size=.5,stratify=y,random_state=SEED)
    classifier=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=7,min_samples_leaf=40,
        l2_regularization=1.,early_stopping=False,random_state=SEED)
    classifier.fit(x[fit],y[fit])
    prediction=classifier.predict_proba(x[test])[:,1]
    return dict(test_auc=float(roc_auc_score(y[test],prediction)),fit_count=len(fit),test_count=len(test),
        features=x.shape[1],id_or_cohort_features_used=False,
        purpose='Statistical domain diagnostic only; no recovery-model update')


def main():
    tick=time.perf_counter();rng=np.random.default_rng(SEED)
    state('loading_existing_arrays')
    with np.load(ROOT/'nikon_abs_three_radii_10ep_20261025/data/metadata.npz') as z:
        data={k:z[k].copy() for k in z.files}
    q=data['q'].astype(np.float64);ids=data['ids'];train=data['train'].astype(bool)
    original=train&(ids<3100000000);added=train&~original;val=~train
    tr=np.flatnonzero(train);orig=np.flatnonzero(original);vi=np.flatnonzero(val)
    body=np.max(np.abs(q.reshape(-1,7,5)),axis=2)>1e-10
    active=body.sum(1);mask_code=(body*(1<<np.arange(7))).sum(1)
    rms=np.sqrt(np.mean(q*q,axis=1));linf=np.max(np.abs(q),axis=1)
    ids_lookup={int(identity):row for row,identity in enumerate(ids)}
    with np.load(TRIAL/'evaluation6000/final_validation6000.npz') as z:
        pred_lookup={int(identity):row for row,identity in enumerate(z['original_ids'])}
        order=np.asarray([pred_lookup[int(identity)] for identity in ids[vi]])
        fields=z['fields'][order];valid=z['valid'][order]
        command=z['command'][order]
    passed=valid&(fields.max(1)<.07)
    with np.load(ROOT/'nikon_large_debug_20261025/trajectory_4ep/validation_003.npz') as z:
        old_lookup={int(identity):row for row,identity in enumerate(z['original_ids'])}
        order=np.asarray([old_lookup[int(identity)] for identity in ids[vi]])
        old=z['valid'][order]&(z['fields'][order].max(1)<.07)
    pools={'original_train':original,'added_train':added,'all_train':train,'validation':val}
    cohort_counts={name:{str(int(c)):int(np.sum(part&(data['cohort']==c)))
                        for c in np.unique(data['cohort'][part])} for name,part in pools.items()}
    patterns=[]
    for code in np.unique(mask_code):
        selected=mask_code==code;vm=selected[vi]
        patterns.append(dict(code=int(code),bodies=(np.flatnonzero(body[np.flatnonzero(selected)[0]])+1).tolist(),
            active=int(active[np.flatnonzero(selected)[0]]),
            **{name:int(np.sum(part&selected)) for name,part in pools.items()},
            validation_passes=int(passed[vm].sum()),validation_rate=float(passed[vm].mean()) if vm.any() else None))
    activity=[]
    for n in sorted(np.unique(active)):
        vm=active[vi]==n
        activity.append(dict(active=int(n),**{name:int(np.sum(part&(active==n))) for name,part in pools.items()},
            **rate(vm,passed),old_passes=int(old[vm].sum())))
    def tv(first,second):
        return float(sum(abs(np.mean(mask_code[first]==code)-np.mean(mask_code[second]==code))
                         for code in np.unique(mask_code))/2)
    coordinate=[]
    for j in range(35):
        coordinate.append(dict(body=j//5+1,dof=('dx','dy','dz','tx','ty')[j%5],
            original_vs_validation=compare(q[original,j],q[val,j]),
            all_train_vs_validation=compare(q[train,j],q[val,j])))
    lower=q[train].min(0);upper=q[train].max(0)
    outside=(q[val]<lower)|(q[val]>upper)
    margin=np.maximum(lower-q[val],q[val]-upper).clip(min=0)
    report=dict(scope=dict(training=28000,original_training=24000,added_training=4000,validation=6000,
        passes=int(passed.sum()),old_passes=int(old.sum()),new_optical_traces=0,
        recovery_network_updates=0,recovery_network_inferences=0),
        generator=dict(original='Same joint_context generator and independent Uniform[-1,1] active coordinates',
            split='Per cohort batches0..39 and50..89 train;40..49 and90..99 validation',
            new4000='2000 all7,1000 first6,500 bodies1/2/3/7,500 bodies4/5/6/7; training only'),
        cohort_counts=cohort_counts,activity=activity,patterns=patterns,
        active_pattern_total_variation=dict(original_vs_validation=tv(original,val),all_train_vs_validation=tv(train,val)),
        validation_unseen_pattern_cases=int(sum(p['validation'] for p in patterns if p['all_train']==0)),
        coordinate_marginals=coordinate,
        validation_outside_train_coordinate_envelope=dict(cases=int(outside.any(1).sum()),
            coordinate_entries=int(outside.sum()),max_excess_normalized_stroke=float(margin.max())),
        pose_rms=dict(original_vs_validation=compare(rms[original],rms[val]),
                      all_train_vs_validation=compare(rms[train],rms[val])),
        pose_linf=dict(original_vs_validation=compare(linf[original],linf[val])))
    save('pose_distribution.json',report)
    state('pose_neighborhoods',unseen_pattern_cases=report['validation_unseen_pattern_cases'])
    # Match original-training pseudo-heldout queries to validation mask counts.
    calibration=[];dval=np.empty(len(vi));dval_old=np.empty(len(vi));dval5=np.empty(len(vi))
    query_active=[];dcal=[];dcal5=[];rank_in_pattern=np.empty(len(vi));benefit=np.empty(len(vi))
    for code in np.unique(mask_code[vi]):
        train_rows=np.flatnonzero(train&(mask_code==code));old_rows=np.flatnonzero(original&(mask_code==code))
        query=np.flatnonzero(mask_code[vi]==code);vrows=vi[query]
        count=min(len(vrows),len(old_rows));cal=rng.choice(old_rows,count,replace=False)
        dims=np.flatnonzero(np.repeat(body[vrows[0]],5));scale=np.sqrt(len(dims))
        tree=cKDTree(q[train_rows][:,dims]);old_tree=cKDTree(q[old_rows][:,dims])
        dv,index=tree.query(q[vrows][:,dims],k=5,workers=4)
        dc,ci=tree.query(q[cal][:,dims],k=6,workers=4)
        # First neighbor is the exact calibration row; exclude it.
        dc=dc[:,1:];old_distance=old_tree.query(q[vrows][:,dims],k=1,workers=4)[0]
        dval[query]=dv[:,0]/scale;dval5[query]=dv[:,-1]/scale;dval_old[query]=old_distance/scale
        dcal.extend(dc[:,0]/scale);dcal5.extend(dc[:,-1]/scale)
        calibration.extend(cal);query_active.extend([len(dims)//5]*len(cal))
        rank_in_pattern[query]=np.searchsorted(np.sort(dc[:,0]/scale),dv[:,0]/scale,side='right')/len(cal)
        benefit[query]=(old_distance-dv[:,0])/np.maximum(old_distance,1e-12)
    calibration=np.asarray(calibration);dcal=np.asarray(dcal);dcal5=np.asarray(dcal5)
    distances=dict(nearest_rms_normalized_coordinate=compare(dcal,dval),fifth_neighbor=compare(dcal5,dval5),
        calibration_count=len(calibration),calibration='Original training queries, self excluded, mask-count matched to validation; same reference training pool',
        validation_above_matched_pattern_calibration_p95=float(np.mean(rank_in_pattern>.95)),
        validation_above_matched_pattern_calibration_p99=float(np.mean(rank_in_pattern>.99)),
        validation_distance_thresholds={str(t):int((dval<t).sum()) for t in (.1,.2,.3,.4,.5)},
        additional4000_relative_nearest_distance_improvement=describe(benefit),
        additional4000_improved_cases=int((benefit>1e-8).sum()),
        failure_auc_pose_distance=float(roc_auc_score(~passed,dval)),
        failure_auc_within_pattern_distance_percentile=float(roc_auc_score(~passed,rank_in_pattern)),
        recovery_by_within_pattern_coverage=binned(rank_in_pattern,passed,old,np.linspace(0,1,6)),
        by_activity=[])
    for n in np.unique(active[vi]):
        take=active[vi]==n;c=np.asarray(query_active)==n
        distances['by_activity'].append(dict(active=int(n),validation=describe(dval[take]),
            training_pseudoheldout=describe(dcal[c]),**rate(take,passed),
            failure_distance_spearman=float(spearmanr(dval[take],(~passed[take]).astype(float)).statistic)))
    report['pose_neighborhoods']=distances
    report['recovery_by_pose_rms']=binned(rms[vi],passed,old,
        np.r_[-np.inf,np.quantile(rms[original],[.2,.4,.6,.8]),np.inf])
    save('pose_coverage.json',report)
    np.savez(HERE/'pose_coverage_arrays.npz',validation_ids=ids[vi],distance=dval,distance_original_train=dval_old,
        calibration_ids=ids[calibration],calibration_distance=dcal,coverage_percentile=rank_in_pattern,
        passed=passed,old_passed=old,fields=fields,active_bodies=active[vi],pose_rms=rms[vi])
    state('loading_observations',pose_distance_ratio=float(np.median(dval)/np.median(dcal)))
    raw=np.empty((len(ids),147),dtype=np.float32)
    files=json.loads((ROOT/'nikon_abs_wide_10ep_20261025/data/sources.json').read_text())['files']
    for file in files:
        with np.load(file) as z:
            block_ids=z['seeds'] if 'seeds' in z.files else z['ids']
            rows=np.asarray([ids_lookup[int(identity)] for identity in block_ids])
            raw[rows]=z['raw'].reshape(len(rows),147)
    extra=np.load(TRIAL/'data/extra.npy',mmap_mode='r')
    signal=np.c_[raw,extra].astype(np.float64)
    mean=signal[train].mean(0);std=np.maximum(signal[train].std(0),.015)
    standardized=(signal-mean)/std
    covariance=standardized[train].T@standardized[train]/len(tr)
    eig,vec=np.linalg.eigh(covariance);order=np.argsort(eig)[::-1];eig=eig[order];vec=vec[:,order]
    keep=32;floor=eig[0]*1e-6
    pca=(standardized@vec[:,:keep]/np.sqrt(np.maximum(eig[:keep],floor))).astype(np.float32)
    pca_variance=float(eig[:keep].sum()/eig.sum())
    means=(signal[val].mean(0)-signal[original].mean(0))/np.maximum(signal[original].std(0),.015)
    rms_signal=np.c_[np.sqrt(np.mean(raw.reshape(-1,3,49)**2,axis=2)),
        np.sqrt(np.mean(extra[:,:197]**2,axis=1)),
        np.sqrt(np.mean(extra[:,197:].reshape(-1,4,49)**2,axis=2))]
    names=['native_center_DIFF','minus_0p05_DIFF','plus_0p05_DIFF','center_ABS',
           'minus_0p1_DIFF','plus_0p1_DIFF','minus_0p15_DIFF','plus_0p15_DIFF']
    obs=dict(features=540,pca_components=32,pca_variance_fraction=pca_variance,
        pca_scaling='Train-only node standardization then whiten leading32PCs; eigen floor1e-6 of maximum',
        original_vs_validation_max_absolute_node_standardized_mean_difference=float(np.max(np.abs(means))),
        original_vs_validation_median_absolute_node_standardized_mean_difference=float(np.median(np.abs(means))),
        channel_rms={name:dict(original_vs_validation=compare(rms_signal[original,i],rms_signal[val,i]),
                              all_train_vs_validation=compare(rms_signal[train,i],rms_signal[val,i]))
                     for i,name in enumerate(names)})
    state('observation_neighborhoods',pca_variance=pca_variance)
    neighbors=NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean',n_jobs=4).fit(pca[train])
    vd,vn=neighbors.kneighbors(pca[val]);cd,cn=neighbors.kneighbors(pca[calibration])
    obs_vd=vd[:,0];obs_cd=cd[:,1]
    obs['neighbors']=dict(distance=compare(obs_cd,obs_vd),
        calibration='Same mask-count-matched original train queries, exact self excluded',
        above_calibration_p95=float(np.mean(obs_vd>np.quantile(obs_cd,.95))),
        above_calibration_p99=float(np.mean(obs_vd>np.quantile(obs_cd,.99))),
        nearest_neighbor_from_added4000_rate=float(added[tr[vn[:,0]]].mean()),
        failure_auc_distance=float(roc_auc_score(~passed,obs_vd)),
        recovery_by_coverage=binned(obs_vd,passed,old,
            np.r_[-np.inf,np.quantile(obs_cd,[.2,.4,.6,.8]),np.inf]))
    native_rms=np.sqrt(np.mean(raw*raw,axis=1))
    obs['recovery_by_native_signal_amplitude']=binned(native_rms[vi],passed,old,
        np.r_[-np.inf,np.quantile(native_rms[original],[.2,.4,.6,.8]),np.inf])
    report['observations']=obs
    state('domain_separation_statistics')
    qfeatures=np.c_[q,body.astype(float),rms,linf]
    report['domain_classifiers']={
        'pose_original_train_vs_validation':domain_auc(qfeatures,orig,vi),
        'pose_all_train_vs_validation':domain_auc(qfeatures,tr,vi),
        'observation_original_train_vs_validation':domain_auc(pca,orig,vi),
        'observation_all_train_vs_validation':domain_auc(pca,tr,vi)}
    report['cost']=dict(active_wall_seconds=time.perf_counter()-tick,cpu_only=True,
        neural_training=False,optical_inference=False,new_observations_generated=False,
        fitted_diagnostic_domain_classifiers=4)
    save('coverage.json',report)
    np.savez(HERE/'observation_coverage_arrays.npz',validation_ids=ids[vi],distance=obs_vd,
             calibration_distance=obs_cd,native_signal_rms=native_rms[vi],passed=passed)
    with (HERE/'validation_cases.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['id','cohort','active_bodies','pose_rms','pose_nearest_distance',
            'within_pattern_coverage_percentile','observation_nearest_distance','native_signal_rms',
            'center','edge_minus','edge_plus','pass','old_pass'])
        for i,row in enumerate(vi):
            writer.writerow([int(ids[row]),int(data['cohort'][row]),int(active[row]),rms[row],dval[i],
                rank_in_pattern[i],obs_vd[i],native_rms[row],*fields[i],int(passed[i]),int(old[i])])
    state('completed',active_seconds=time.perf_counter()-tick,
        unseen_pattern_cases=report['validation_unseen_pattern_cases'],
        pose_distance_ratio=float(np.median(dval)/np.median(dcal)),
        observation_distance_ratio=float(np.median(obs_vd)/np.median(obs_cd)),
        domain_auc={k:v['test_auc'] for k,v in report['domain_classifiers'].items()})


if __name__=='__main__':
    try:
        with threadpool_limits(limits=4):
            main()
    except BaseException:
        state('failed',error=traceback.format_exc());raise
