"""Four real FP32 network-training epochs, original samples and architecture.

Full camera finite differences are NOT evaluated inside the training batches.
Only cached images, supervised poses and the existing differentiable wavefront
physics are used. Final original500 evaluates the neural candidate separately.
"""
from local import *
from training_objective import FullFieldObjective
from concurrent.futures import ThreadPoolExecutor
import math,gc,traceback,ctypes

OUT=RUN/'training';EPOCHS=4;SEED=2026102609


def summary(values,valid):
    good=valid&np.isfinite(values).all(1)
    return dict(count=len(values),center_pass=int((good&(values[:,0]<.07)).sum()),
        fullfield_pass=int((good&(values<.07).all(1)).sum()),invalid=int((~good).sum()),
        valid_center_mean=float(values[good,0].mean()) if good.any() else None,
        valid_edge_diff_mean=float(values[good,1:].mean()) if good.any() else None)


@torch.no_grad()
def evaluate(model,blocks):
    directory=RUN/'evaluation500';directory.mkdir(exist_ok=True)
    if (directory/'result.json').exists():return json.loads((directory/'result.json').read_text())
    metrics=DiffMetrics()
    with np.load(SOURCE/'training/selected_validation.npz') as z:baseline={k:z[k].copy() for k in z.files}
    old_saved={}
    for f in (LEGACY/'evaluation500').glob('case_*.json'):
        r=json.loads(f.read_text());old_saved[r['case']]=r['baseline']
    records=[]
    for b in range(280,300):
        path=directory/f'batch_{b}.npz'
        if path.exists():
            with np.load(path) as z:records.append({k:z[k].copy() for k in z.files})
            continue
        x,q=blocks.read('joint',b);command=[];fields=[];valid=[]
        for s in range(0,25,5):
            cmd=model.command(torch.as_tensor(x[s:s+5].astype(np.float32),device='cuda')).cpu().numpy()
            v,g=metrics.score(q[s:s+5],cmd);command.append(cmd);fields.append(v);valid.append(g)
        index=np.arange(b*25,(b+1)*25);oldfields=[];oldvalid=[]
        missing=[i for i in range(25) if int(index[i]) not in old_saved]
        measured={}
        if missing:
            slots=[int(np.where(baseline['index']==index[i])[0][0]) for i in missing]
            v,g=metrics.score(baseline['initial'][slots],baseline['command'][slots])
            measured={i:(vv,gg) for i,vv,gg in zip(missing,v,g)}
        for i,case in enumerate(index):
            if int(case) in old_saved:r=old_saved[int(case)];oldfields.append(r['mixed_wrms']);oldvalid.append(r['valid'])
            else:v,g=measured[i];oldfields.append(v);oldvalid.append(g)
        rec=dict(index=index,command=np.concatenate(command),mixed_wrms=np.concatenate(fields),valid=np.concatenate(valid),
                 baseline_mixed=np.array(oldfields),baseline_valid=np.array(oldvalid))
        np.savez_compressed(path,**rec);records.append(rec)
        state('network_evaluation500',completed=(b-279)*25,total=500)
    arrays={k:np.concatenate([r[k] for r in records]) for k in records[0]}
    np.savez_compressed(directory/'predictions.npz',**arrays)
    before=arrays['baseline_valid']&(arrays['baseline_mixed']<.07).all(1)
    after=arrays['valid']&(arrays['mixed_wrms']<.07).all(1)
    result=dict(baseline=summary(arrays['baseline_mixed'],arrays['baseline_valid']),
        candidate=summary(arrays['mixed_wrms'],arrays['valid']),gained=int((~before&after).sum()),lost=int((before&~after).sum()),
        neural_forward_only=True,model_refinement=False,acquisition_packets=1,truth_used_only_in_scoring=True,
        independent_review=False,automatic_deployment=False,assistant_reviewed=False)
    common.atomic(directory/'result.json',result);print('RESULT '+json.dumps(result),flush=True);return result


def run():
    OUT.mkdir(exist_ok=True)
    gate=json.loads((RUN/'repair_pilot/result.json').read_text())
    if not gate['gate_pass']:raise RuntimeError('Repair pilot failed its fixed local validity/retention gate; do not train automatically')
    if (OUT/'complete.json').exists():return
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    resume=OUT/'resume.pt'
    model,payload=restore(resume if resume.exists() else SOURCE/'training/candidate.pt')
    assert all(p.dtype==torch.float32 for p in model.parameters())
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
    ep0=1;first=0;updates=0;history=[];active=0.;micro=25;loss_sum=0.;seen=0
    if resume.exists():
        opt.load_state_dict(payload['optimizer']);ep0=payload['next_epoch'];first=payload['next_block'];updates=payload['updates']
        history=payload['history'];active=payload['active'];micro=payload['micro'];loss_sum=payload['loss_sum'];seen=payload['seen']
        torch.set_rng_state(payload['rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
    cal=payload['calibrations'];blocks=Blocks(capacity=4);obj=FullFieldObjective()
    def checkpoint(ep,block):
        save(resume,dict(model=model.state_dict(),calibrations=cal,optimizer=opt.state_dict(),next_epoch=ep,next_block=block,
            updates=updates,history=history,active=active,micro=micro,loss_sum=loss_sum,seen=seen,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    common.atomic(OUT/'configuration.json',dict(epochs=4,start_checkpoint=str(SOURCE/'training/candidate.pt'),start_epoch=45,
        network_parameters=sum(p.numel() for p in model.parameters()),parameter_dtype='float32',
        data='Original7000 joint plus7000 isolated rehearsal per epoch; original alternating front/rear schedule',
        optimizer='AdamW all parameters1e-5 cosine to2e-6; weight_decay1e-4; gradient_clip5',
        loss='0.5 centerABS +0.5 mean8 edgeDIFF bounded physical loss;0.2 raw-pose SmoothL1 beta.25;10 soft domain penalty beyond.95',
        inference='Same single noisy nine-field packet; raw35 prediction clipped to original[-1,1] then negated',
        physical_dtype='Existing float64 ray trace only; neural network and optimizer FP32',
        effective_batch=25,microbatch=micro,full_resolution_images='Reuse original2048pupil/4096FFT generated packets; no recapture or per-batch FFT',
        final_evaluation='Original500 once, centerABS plus8edgeDIFF<.07; old-center criterion also reported',
        repaired_inverse='Separately tested diagnostic; no claim this is included in neural-only500 scores',
        independent_review=False,digest_comparisons=False,automatic_deployment=False))
    state('training_initializing',epoch=ep0,epochs=4,updates=updates,network_parameters=sum(p.numel() for p in model.parameters()))
    for ep in range(ep0,5):
        lr=2e-6+.5*(1e-5-2e-6)*(1+math.cos(math.pi*(ep-1)/4))
        for group in opt.param_groups:group['lr']=lr
        order=np.random.default_rng(SEED+ep).permutation(280);branch='front' if ep%2 else 'rear'
        schedule=[pair for b in order for pair in [('joint',int(b)),(branch,int(b))]]
        epoch_tick=common.clock()
        # One bounded I/O prefetch; GPU work remains serial.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(blocks.read,*schedule[first]) if first<len(schedule) else None
            for block in range(first,len(schedule)):
                tick=common.clock();x,q=future.result()
                future=pool.submit(blocks.read,*schedule[block+1]) if block+1<len(schedule) else None
                model.train()
                while True:
                    opt.zero_grad(set_to_none=True);loss_value=0.;parts=[]
                    try:
                        for start in range(0,25,micro):
                            xx=torch.as_tensor(x[start:start+micro].astype(np.float32),device='cuda')
                            qq=torch.as_tensor(q[start:start+micro],device='cuda').double()
                            loss,p=obj.loss(model(xx),qq)
                            if not bool(torch.isfinite(loss)):raise RuntimeError(f'Nonfinite loss epoch{ep} block{block}')
                            (loss*len(xx)/25).backward();loss_value+=float(loss.detach())*len(xx);parts.append(p)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step();break
                    except torch.cuda.OutOfMemoryError:
                        opt.zero_grad(set_to_none=True)
                        if micro<=1:raise
                        micro=max(1,micro//2);xx=None;qq=None;loss=None;p=None;gc.collect();torch.cuda.empty_cache()
                updates+=1;loss_sum+=loss_value;seen+=25;active+=common.clock()-tick
                if block==first or (block+1)%10==0 or block==559:
                    checkpoint(ep,block+1)
                    remaining=(4-ep)*560+559-block
                    state('training',epoch=ep,epochs=4,block=block+1,blocks=560,updates=updates,loss=loss_sum/seen,
                        parts=parts[-1],microbatch=micro,active_seconds=active,
                        mean_seconds_per_update=active/updates,estimated_remaining_train_seconds=remaining*active/updates)
                    print(json.dumps(dict(epoch=ep,block=block+1,updates=updates,loss=loss_sum/seen,active_seconds=active)),flush=True)
        history.append(dict(epoch=ep,loss=loss_sum/seen,epoch_active_seconds=common.clock()-epoch_tick,updates=updates))
        common.atomic(OUT/'history.json',history)
        save(OUT/f'epoch_{ep}.pt',dict(model=model.state_dict(),calibrations=cal,epoch=ep,updates=updates))
        loss_sum=0.;seen=0;first=0;checkpoint(ep+1,0)
    save(OUT/'last.pt',dict(model=model.state_dict(),calibrations=cal,epoch=4,updates=updates))
    model.eval();evaluation=evaluate(model,blocks)
    common.atomic(OUT/'complete.json',dict(epochs=4,updates=updates,active_training_seconds=active,evaluation=evaluation,
        assistant_reviewed=False,completed=common.stamp()))
    state('completed',epochs=4,updates=updates,result=str(RUN/'evaluation500/result.json'),assistant_reviewed=False)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    try:run()
    except BaseException:state('failed',error=traceback.format_exc());raise
