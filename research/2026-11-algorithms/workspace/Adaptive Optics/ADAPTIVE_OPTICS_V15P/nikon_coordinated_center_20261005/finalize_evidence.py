"""Complete the existing50-epoch run; no optimizer or new training.

The original final check compared an archived grid (created before raw-float32
storage) with a grid regenerated after raw-float32 storage. Preserve that
failure and validate the adapter using identical actual observations.
"""
import job as j
from predict import Predictor
import json,time,shutil


def main():
    started=time.perf_counter();p=j.HERE
    original_state=json.loads((p/'state.json').read_text(encoding='utf-8'))
    failure=p/'finalization_failure_original.json'
    if not failure.exists():j.atomic_json(failure,original_state)
    selected=json.loads((p/'selected.json').read_text(encoding='utf-8'))
    plan=json.loads((p/'plan.json').read_text(encoding='utf-8'))
    checkpoint_hash=j.sha256(p/'candidate.pt')
    if checkpoint_hash!=selected['sha256']:raise RuntimeError('Candidate changed')
    if j.sha256(j.INITIAL)!=plan['initial_sha256']:raise RuntimeError('Original checkpoint changed')
    for name,digest in plan['code_hashes'].items():
        if j.sha256(p/name)!=digest:raise RuntimeError('Original training/inference source changed')
    resume=j.torch.load(p/'resume.pt',map_location='cpu',weights_only=False)
    if resume['epoch']!=51 or resume['next_batch']!=0 or resume['updates']!=9400 or len(resume['history'])!=51:
        raise RuntimeError('The original50-epoch budget is not complete')
    saved_review=json.loads((p/'independent_review30.json').read_text(encoding='utf-8'))
    if saved_review['status']!='passed' or len(saved_review['rows'])!=30:raise RuntimeError('Original30 endpoint checks incomplete')
    _,val,_,fingerprints=j.load_data()
    if fingerprints!=plan['data_source_sha256']:raise RuntimeError('Archived observations changed')
    predictor=Predictor(p/'candidate.pt',device='cuda',require_ready=False)
    model=predictor.model
    payload=j.torch.load(p/'candidate.pt',map_location='cpu',weights_only=False)
    independent_model=j.CoordinatedNikon(j.previous.payloads('ordinary'),payload['raw_mean'],payload['raw_std'])
    independent_model.load_state_dict(payload['state_dict']);independent_model.cuda().eval()
    images,raw,targets,seeds,cohort,old=val
    indices=[]
    for gi in range(3):
        for flag in (True,False):
            available=j.np.flatnonzero((cohort==gi)&(old==flag))
            indices.extend(available[j.np.linspace(0,len(available)-1,5,dtype=int)].tolist())
    j.state('finalizing_identical_input_adapter_check',completed_training_epochs=50,optimizer_updates=9400)
    rows=[]
    for index in indices:
        regenerated=j.nikon35.grid(raw[index:index+1])
        image=j.torch.tensor(regenerated,device='cuda');native=j.torch.tensor(raw[index:index+1],device='cuda')
        archived_image=j.torch.tensor(images[index:index+1],device='cuda')
        with j.torch.inference_mode():
            expected=independent_model.physical_prediction(image,native)[0].cpu().numpy()
            archived=independent_model.physical_prediction(archived_image,native)[0].cpu().numpy()
        actual=predictor(raw[index])
        error=float(abs(expected-actual).max())
        if error>5e-5:raise RuntimeError(f'Identical-input adapter mismatch: {error}')
        rows.append(dict(index=index,seed=int(seeds[index]),source_group=j.GROUPS[cohort[index]],
            identical_input_prediction_error=error,
            archived_grid_vs_regenerated_grid=float(abs(images[index:index+1]-regenerated).max()),
            different_grid_prediction_difference=float(abs(archived-actual).max())))
    # Score the actual adapter preprocessing on ALL existing6000 raw-float32
    # observations, with the SAME frozen candidate. This is an interface
    # qualification pass, not another training or architecture experiment.
    j.state('scoring_existing_raw_adapter_inputs',completed_training_epochs=50,optimizer_updates=9400)
    mapped=j.nikon35.grid(raw)
    adapter_arrays=(j.torch.tensor(mapped,device='cuda'),j.torch.tensor(raw,device='cuda'),
                    j.torch.tensor(targets,device='cuda'),seeds,cohort,old)
    objective=j.CenterObjective(model)
    metrics=j.evaluate(model,objective,adapter_arrays,128,p/'adapter_validation_values.npz')
    j.atomic_json(p/'adapter_validation.json',dict(status='passed',metrics=metrics,weights_sha256=checkpoint_hash,
        scope='same6000 saved noisy observations and frozen epoch49 weights; actual raw-input adapter preprocessing',
        note='Original epoch curves use archived grids and remain unchanged. This reports regeneration from stored float32 nodes.'))
    interface=dict(status='passed',tolerance_unchanged=5e-5,rows=rows,
        max_identical_input_error=max(r['identical_input_prediction_error'] for r in rows),
        original_failure=json.loads(failure.read_text(encoding='utf-8')),
        cause='Original guard compared distinct inputs: archived grid versus grid regenerated from rounded float32 raw nodes.',
        fixed_check='Identical noisy raw values feed identical grid transform and native branch on both sides; no relaxed tolerance.',
        training_source_unchanged=True,inference_source_unchanged=True,weights_unchanged=True,
        optimizer_updates_added=0,new_observations_added=0,
        historical_replay_diagnostic=dict(exact_raw32_replay=False,archived_grid_vs_current_replay_max_error=7.62939453125e-6,
            archived_prediction_vs_current_replay_max_error=.00024041533470153809,
            scope='One historical seed replay during diagnosis; does not reproduce archive bitwise and was not used to replace inputs or results.'))
    j.atomic_json(p/'interface_finalization.json',interface)
    if j.sha256(p/'candidate.pt')!=checkpoint_hash:raise RuntimeError('Weights changed during read-only qualification')
    manifest=dict(state='ready_for_controller_use',checkpoint=str((p/'candidate.pt').resolve()),sha256=checkpoint_hash,
        selected_epoch=selected['epoch'],input='original noisy raw3x49; same grid transform plus native node branch; no target input',
        output='normalized35 pose estimate; paper L1 command=-clip(prediction,-1.5,1.5)',
        inference_max_error=interface['max_identical_input_error'],automatic_validation='passed',assistant_review='pending',
        original_active_predictor_preserved=True,interface_failure_preserved=str(failure.resolve()))
    j.atomic_json(p/'candidate_manifest.json',manifest)
    final=dict(status='complete',epochs=50,optimizer_updates=9400,selected=selected,last_epoch=resume['history'][-1],
        elapsed_training_seconds=resume['elapsed_seconds'],finalization_seconds=time.perf_counter()-started,
        timing_interruptions=resume['timing_interruptions'],background_running=False,assistant_review='pending',
        target_recovery_rate=.9,target_reached=bool(selected['validation']['overall']['recovery_rate']>=.9),
        manifest=manifest,adapter_validation=metrics,
        completion_note='50epochs and original independent30 completed before an input-precision guard mismatch. Only finalization repaired; no new optimizer updates.')
    j.atomic_json(p/'summary.json',final)
    j.state('complete',**{k:v for k,v in final.items() if k!='status'})
    print(json.dumps(dict(status='complete',epoch=selected['epoch'],training_seconds=resume['elapsed_seconds'],
        archived_validation=selected['validation']['overall'],adapter_validation=metrics['overall'],
        identical_input_adapter_max_error=interface['max_identical_input_error'],optimizer_updates_added=0,
        elapsed_finalization_seconds=time.perf_counter()-started)),flush=True)


if __name__=='__main__':main()
