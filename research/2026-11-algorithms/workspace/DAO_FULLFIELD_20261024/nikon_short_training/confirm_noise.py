"""Matched full6000pose noise confirmation; no training or pose selection."""
from short_train import TrialSystem,HERE,exp,torch,np
import argparse,json,traceback,shutil

p=argparse.ArgumentParser();p.add_argument('--idea',required=True)
p.add_argument('--noise',choices=['original','new'],default='new');a=p.parse_args()
out=HERE/a.idea/('noise_confirmation' if a.noise=='new' else 'original_protocol_comparison');out.mkdir(exist_ok=True)
plan=json.loads((HERE/'confirmation_plan.json').read_text())
assert plan['frozen_before_new_noise_evaluation']
source=HERE/a.idea
assert (source/'training_evidence.json').exists()
try:
    generator=TrialSystem('sensor_consistency')
    original=generator.load_data();validation=np.flatnonzero(~original['train'])
    ids=original['ids'][validation]
    seeds=(ids.astype(np.uint64)+1000000007)%(2**32) if a.noise=='new' else ids.copy()
    q=original['q'][validation]
    if a.noise=='new':
        raw=[]
        with torch.no_grad():
            for start in range(0,len(q),128):
                physical=torch.as_tensor(q[start:start+128],dtype=torch.float64,device='cuda')
                raw.append(generator.native_optics.metrics(physical)['difference'].cpu().numpy())
        raw=np.concatenate(raw)
        for index,seed in enumerate(seeds):
            raw[index]+=np.random.RandomState(int(seed)).normal(0.,.015,raw[index].shape)
        images=generator.legacy.nikon35.grid(raw)
    else:
        raw=original['raw'][validation];images=original['image'][validation]
    data=dict(image=images,raw=raw.astype(np.float32),q=q,
        ids=seeds,cohort=original['cohort'][validation],train=np.zeros(len(q),dtype=bool),files=original['files'])
    del original
    exp.atomic(out/'manifest.json',dict(plan=plan,pose_count=len(q),pose_source_indices=validation.tolist(),
        original_ids=ids.tolist(),noise_seeds=seeds.tolist(),training=False,
        independent_scenes=False,selection_on_this_noise=a.noise=='original',noise_mode=a.noise))
    selected=torch.load(source/'candidate.pt',map_location='cpu',weights_only=False)
    routes=[('original_direct','bounded_control',None,False),('original_measured_loop','closed_loop_training',None,False)]
    if a.idea!='closed_loop_training':
        routes.extend([('idea_epoch0_measured_loop',a.idea,None,True),('idea_epoch0',a.idea,None,False)])
        routes.append(('idea_selected_measured_loop',a.idea,source/'candidate.pt',True))
    routes.append(('idea_selected',a.idea,source/'candidate.pt',False))
    if selected['epoch']!=4:
        if a.idea!='closed_loop_training':routes.append(('idea_epoch4_measured_loop',a.idea,source/'last.pt',True))
        routes.append(('idea_epoch4',a.idea,source/'last.pt',False))
    results={};initializer_equal=None
    for name,idea,checkpoint,force_loop in routes:
        record=out/(name+'.json');arrays=out/(name+'.npz')
        if record.exists() and arrays.exists():results[name]=json.loads(record.read_text());continue
        if int(selected['epoch'])==0 and name.startswith('idea_selected'):
            base_name=name.replace('idea_selected','idea_epoch0',1)
            base_record=out/(base_name+'.json');base_arrays=out/(base_name+'.npz')
            if base_record.exists() and base_arrays.exists():
                if initializer_equal is None:
                    check_system=TrialSystem(a.idea);check_model=check_system.model('large')
                    current=check_model.state_dict()
                    initializer_equal=set(current)==set(selected['model']) and all(
                        torch.equal(value.cpu(),selected['model'][key]) for key,value in current.items())
                    del current,check_model,check_system
                if initializer_equal:
                    result=json.loads(base_record.read_text())
                    result.update(checkpoint=str(source/'candidate.pt'),reused_from=str(base_record),
                        initializer_state_exactly_equal=True)
                    shutil.copyfile(base_arrays,arrays);exp.atomic(record,result);results[name]=result
                    print(name,'reused after exact initializer-state comparison',flush=True)
                    continue
        reference_dir=HERE/'nonlinear_local_model'/out.name
        if name.startswith('original_') and reference_dir!=out:
            old_record=reference_dir/(name+'.json');old_arrays=reference_dir/(name+'.npz')
            if old_record.exists() and old_arrays.exists():
                with np.load(old_arrays) as z:
                    assert np.array_equal(z['original_ids'],seeds) and np.array_equal(z['initial'],q)
                result=json.loads(old_record.read_text());result['reused_from']=str(old_record)
                shutil.copyfile(old_arrays,arrays);exp.atomic(record,result);results[name]=result
                print(name,'reused identical completed reference',flush=True)
                continue
        system=TrialSystem(idea);model=system.model('large')
        epoch=0
        if checkpoint is not None:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['model']);epoch=int(saved['epoch'])
        counts=[];direct_records=[];forward=system.forward
        if force_loop:
            from feedback_training import MeasuredFeedback
            additional_feedback=MeasuredFeedback(system)
        def tracked_forward(model,data,indices):
            result=forward(model,data,indices)
            if force_loop:
                endpoint,initial,command,pose=result
                w,valid,absolute=system.metrics(endpoint,initial)
                direct_records.append((w.cpu().numpy(),valid.cpu().numpy(),absolute.cpu().numpy(),command.cpu().numpy()))
                observed=torch.as_tensor(data['raw'][indices],device=command.device)
                command=additional_feedback.run(initial,command,data['ids'][indices],observed)
                result=(initial+command,initial,command,pose)
                counts.append(additional_feedback.last_counts.copy())
            else:
                counts.append(system.feedback.last_counts.copy() if hasattr(system,'feedback') else np.ones(len(indices),dtype=int))
            return result
        system.forward=tracked_forward
        exp.atomic(out/'state.json',dict(status='evaluating',route=name,updated=exp.stamp()))
        metrics=exp.evaluate(system,model,data,np.arange(len(q)),128,arrays)
        reads=np.concatenate(counts)
        result=dict(validation=metrics,epoch=epoch,checkpoint=str(checkpoint) if checkpoint else str(system.warm),
            inference_protocol=system.trial_config.get('inference_protocol','one acquisition: direct network')+
                (' + original diff_calibrated measured feedback' if force_loop else ''),
            acquisitions_mean=float(reads.mean()),acquisitions_max=int(reads.max()),
            fixed_pose_count=len(q),separate_measurement_noise=a.noise=='new')
        exp.atomic(record,result);results[name]=result
        print(name,json.dumps(dict(full=metrics['full_field_passes'],count=metrics['count'],
            center=metrics['center07_rate'],mean=metrics['field_mean'],epoch=epoch,readings=float(reads.mean()))),flush=True)
        if force_loop:
            direct_name=name.removesuffix('_measured_loop')
            w,v,absolute,command=[np.concatenate([row[k] for row in direct_records]) for k in range(4)]
            direct_metrics=exp.aggregate(w,v,absolute,.07)
            direct_metrics['cohorts']={str(g):exp.aggregate(w[data['cohort']==g],v[data['cohort']==g],
                absolute[data['cohort']==g],.07) for g in np.unique(data['cohort'])}
            np.savez_compressed(out/(direct_name+'.npz'),indices=np.arange(len(q)),original_ids=seeds,
                cohort=data['cohort'],initial=q,command=command,fields=w,valid=v,absolute=absolute)
            direct=dict(validation=direct_metrics,epoch=epoch,checkpoint=result['checkpoint'],
                inference_protocol=system.trial_config.get('inference_protocol','one acquisition: direct network'),
                acquisitions_mean=1.,acquisitions_max=1,fixed_pose_count=len(q),separate_measurement_noise=a.noise=='new',
                recorded_before_feedback=True,source_loop_route=name)
            exp.atomic(out/(direct_name+'.json'),direct);results[direct_name]=direct
            print(direct_name,json.dumps(dict(full=direct_metrics['full_field_passes'],count=len(q),
                center=direct_metrics['center07_rate'],mean=direct_metrics['field_mean'],epoch=epoch,readings=1.)),flush=True)
        del model,system
    exp.atomic(out/'summary.json',dict(results=results,plan=plan,updated=exp.stamp(),assistant_review='pending'))
    exp.atomic(out/'state.json',dict(status='completed',updated=exp.stamp()))
except Exception:
    exp.atomic(out/'state.json',dict(status='failed',updated=exp.stamp(),error=traceback.format_exc()));raise
