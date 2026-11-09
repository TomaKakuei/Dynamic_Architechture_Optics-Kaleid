"""Check shared initialization, forward/backward validity and memory; no updates."""
from unified import *
import gc

def main():
    results={};neural={};refined={}
    common=UnifiedSystem('1x');data=common.load_data()
    rng=np.random.default_rng(202610241)
    ix=np.concatenate([rng.choice(np.flatnonzero(data['train']&(data['cohort']==g)),42 if g<2 else 44,replace=False) for g in range(3)])
    for size in ('1x','2x'):
        system=UnifiedSystem(size)
        torch.manual_seed(202610241);torch.cuda.manual_seed_all(202610241)
        model=system.model('large').eval()
        image=torch.as_tensor(data['image'][ix],device='cuda');raw=torch.as_tensor(data['raw'][ix],device='cuda')
        with torch.no_grad():
            neural[size]=UnifiedTrunk.forward(model,image,raw).cpu().numpy()
            refined[size]=model(image,raw).cpu().numpy()
        optimizer=exp.make_optimizer(model)
        assert sum(len(g['params']) for g in optimizer.param_groups)==len(list(model.parameters()))
        torch.cuda.reset_peak_memory_stats();timings=[]
        for _ in range(2):
            model.train();optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();tick=exp.active_time()
            loss,parts=system.loss(model,data,ix);assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            torch.cuda.synchronize();timings.append(exp.active_time()-tick)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        results[size]=dict(parameters=sum(p.numel() for p in model.parameters()),parameter_tensors=len(list(model.parameters())),
            dtypes=sorted({str(p.dtype) for p in model.parameters()}),forward_backward_seconds=timings,
            peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,optimizer_updates=0,
            loss=float(loss.detach()),training_details=parts,optimizer_learning_rates=[g['lr'] for g in optimizer.param_groups if g['params']],
            grouped_neural_parameters=any(k.startswith('base.') for k,_ in model.named_parameters()))
        assert not results[size]['grouped_neural_parameters']
        del model,optimizer,system,loss;gc.collect();torch.cuda.empty_cache()
    neural_difference=float(np.max(np.abs(neural['1x']-neural['2x'])))
    refined_difference=float(np.max(np.abs(refined['1x']-refined['2x'])))
    assert neural_difference<1e-5
    per_epoch={k:float(np.mean(v['forward_backward_seconds'])*188+60) for k,v in results.items()}
    exp.atomic(HERE/'preflight.json',dict(passed=True,updated=exp.stamp(),training_indices=ix.tolist(),results=results,
        initial_neural_max_difference=neural_difference,initial_refined_max_difference=refined_difference,
        expected_seconds_per_epoch=per_epoch,estimate_total_seconds=sum(per_epoch.values())*4+600,
        optimizer_updates=0,scope='128training samples; two discarded forward/backward probes per size; no parameter updates'))
    print(json.dumps(dict(results=results,neural_difference=neural_difference,refined_difference=refined_difference,
        expected_seconds_per_epoch=per_epoch),indent=2))

if __name__=='__main__':main()
