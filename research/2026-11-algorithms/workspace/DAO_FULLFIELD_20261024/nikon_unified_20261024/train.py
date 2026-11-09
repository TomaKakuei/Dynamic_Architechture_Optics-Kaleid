"""One explicitly authorized unified-network size, exactly4full epochs."""
from unified import *
import argparse,ctypes,os,traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('--size',choices=['1x','2x'],required=True);a=p.parse_args()
    assert json.loads((HERE/'preflight.json').read_text())['passed']
    folder=HERE/('unified_'+a.size);folder.mkdir(exist_ok=True)
    args=argparse.Namespace(stage='train',system='nikon',variant='large',epochs=4,batch=128,microbatch=128,seed=202610241)
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        system=UnifiedSystem(a.size)
        exp.train(system,args,out=folder)
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        resume=torch.load(folder/'resume.pt',map_location='cpu',weights_only=False)
        torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        initial=system.model('large')
        deltas={k:float((last['model'][k].double()-v.detach().cpu().double()).square().sum()) for k,v in initial.named_parameters()}
        steps=sorted({int(v['step']) for v in resume['optimizer']['state'].values()})
        assert steps==[752] and sum(deltas.values())>0
        exp.atomic(folder/'training_evidence.json',dict(completed_epochs=4,optimizer_updates=752,
            saved_optimizer_steps=steps,trainable_parameters=sum(p.numel() for p in initial.parameters()),
            parameter_tensors=len(deltas),changed_parameter_tensors=sum(x>0 for x in deltas.values()),
            parameter_only_delta_l2=float(np.sqrt(sum(deltas.values()))),excludes_buffers=True,
            network_dtype='float32',source_checkpoint=str(SOURCE),updated=exp.stamp()))
    except Exception:
        exp.atomic(folder/'failure.json',dict(error=traceback.format_exc(),updated=exp.stamp()));raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
