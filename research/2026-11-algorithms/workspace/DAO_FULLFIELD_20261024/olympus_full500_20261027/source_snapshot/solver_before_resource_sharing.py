from pathlib import Path
import sys
TASK=Path(__file__).resolve().parent
OLD=TASK.parent/'olympus_prediction_loop_20261026'
sys.path.insert(0,str(OLD))
from context import *
from qp import solve_local


class ConstrainedRefiner(Refiner):
    def __init__(self,tag,fd='legacy',rank=8):
        super().__init__('hybrid')
        self.tag=tag;self.fd=fd;self.rank=rank
        self.own_cache=TASK/'responses'/tag;self.own_cache.mkdir(parents=True,exist_ok=True)
        self.inv_cov=np.linalg.inv(self.cov)/128**2

    @torch.no_grad()
    def current_response(self,x,image,key,cycle):
        if self.fd=='legacy':
            oldkey=f'{key}_0_0.2_0.01'
            if cycle==0 and self.rank==8 and (OLD/'loop_responses'/f'{oldkey}.npz').exists():
                self.cache_dir=OLD/'loop_responses'
                return self.response(x,oldkey,self.rank)
            self.cache_dir=self.own_cache;name=f'{key}_{cycle}'
            dest=self.cache_dir/f'{name}.npz'
            if not dest.exists():np.savez_compressed(dest,prior=x,image=image.cpu().numpy())
            return self.response(x,name,self.rank)
        dest=self.own_cache/f'{key}_{cycle}.npz';known={};calls=0
        if dest.exists():
            with np.load(dest) as z:known={k:z[k].copy() for k in z.files}
            if not np.array_equal(known['prior'],x):raise ValueError('Response cache estimate mismatch')
        else:
            seed=TASK/'responses'/'development_j_central8'/f'{key}_0.npz'
            if self.tag!='development_j_central8':
                seed=TASK/'responses'/'development_j_central8'/f'{key}_{cycle}.npz'
            if self.tag!='development_j_central8' and seed.exists():
                with np.load(seed) as z:
                    if np.array_equal(z['prior'],x):known={k:z[k].copy() for k in z.files}
            if not known:known=dict(prior=x,image=image.cpu().numpy(),basis=self.hybrid_basis(x))
        columns=[];hs=[];B=known['basis'][:,:self.rank]
        for k,direction in enumerate(B.T):
            label=f'jac_{k}'
            if label not in known:
                mask=np.abs(direction)>1e-12
                room=(1-np.abs(x[mask]))/np.abs(direction[mask])
                h=min(.005,.9*float(room.min()))
                if h<1e-6:raise ValueError('Insufficient symmetric finite-difference space')
                plus,_=self.camera.evaluate(x+h*direction);minus,_=self.camera.evaluate(x-h*direction)
                calls+=2;known[label]=((plus-minus)/(2*h)).cpu().numpy();known[f'h_{k}']=h
                np.savez_compressed(dest,**known)
            columns.append(torch.as_tensor(known[label],device='cuda'));hs.append(float(known[f'h_{k}']))
        return image,torch.stack(columns,dim=-1),B,calls,hs

    @torch.no_grad()
    def predict(self,observed,prior,start,key,cycles=3,radius=.2,center_limit=.01,adaptive=False):
        observed=torch.as_tensor(observed,device='cuda').float();sigma=sigma_from_adu(observed)
        prior=np.asarray(prior,dtype=np.float64);x=np.asarray(start,dtype=np.float64).copy()
        P=torch.as_tensor(self.inv_cov,device='cuda').double();calls=0;history=[];tick=common.clock()
        initial_center_limit=center_limit
        def render(q):
            nonlocal calls
            calls+=1
            return self.camera.evaluate(q)[0]
        def merit(image,q):
            d=torch.as_tensor(q-prior,device='cuda').double()
            return float((((image-observed)/sigma/128).square().sum())+d@P@d)
        cached=OLD/'model_images'/f'initial_{key}.npz'
        try:
            if cached.exists():
                with np.load(cached) as z:
                    if not np.array_equal(z['estimate'],x):raise ValueError('Initial cache mismatch')
                    current=torch.as_tensor(z['image'],device='cuda')
            else:current=render(x)
            cost=merit(current,x);initial=cost
            for cycle in range(cycles):
                if cost<=9:history.append(dict(cycle=cycle,action='noise_stop',cost=cost));break
                prediction,j,B,ncalls,hs=self.current_response(x,current,key,cycle);calls+=ncalls
                J=(j/sigma[...,None]/128).flatten(0,-2).double()
                r=((prediction-observed)/sigma/128).flatten().double();bt=torch.as_tensor(B,device='cuda').double()
                d=torch.as_tensor(x-prior,device='cuda').double()
                Q=J.T@J+bt.T@P@bt;g=J.T@r+bt.T@P@d
                dz,info=solve_local(Q.cpu().numpy(),g.cpu().numpy(),B,x,prior,self.center_gram,radius,center_limit)
                before=cost;trials=[];accepted=False;accepted_rho=None;used_center_limit=center_limit
                if info['feasible'] and info['predicted_gain']>1e-8:
                    delta=B@dz
                    for gain in (1.,.5,.25):
                        q=x+gain*delta
                        if not self.camera.camera.engine.geometry.path(q)['safe']:continue
                        if not self.camera.camera.engine.geometry.path(prior-q,start=prior)['safe']:continue
                        try:im=render(q);value=merit(im,q)
                        except (ValueError,RuntimeError) as error:
                            trials.append(dict(gain=gain,error=str(error)));continue
                        u=torch.as_tensor(gain*dz,device='cuda').double()
                        predicted=float((r+J@u).square().sum()+(d+bt@u)@P@(d+bt@u))
                        rho=(before-value)/max(before-predicted,1e-12)
                        trials.append(dict(gain=gain,cost=value,predicted_cost=predicted,agreement=rho))
                        if value<before and rho>.05:
                            x=q;current=im;cost=value;accepted=True;accepted_rho=rho;break
                center_change=float(np.sqrt(max((x-prior)@self.center_gram@(x-prior),0)))
                if adaptive and accepted and cost>9 and accepted_rho>=.75 and center_change>=.9*center_limit:
                    center_limit=min(.04,center_limit*2)
                history.append(dict(cycle=cycle,before=before,after=cost,accepted=accepted,solver=info,trials=trials,
                    finite_difference_steps=hs,nominal_center_change=center_change,
                    center_limit=used_center_limit,next_center_limit=center_limit))
                if not accepted:break
            return dict(command=-x,steps=history,actual_camera_calls=calls,network_calls=0,
                initial_image_cost=initial,final_image_cost=cost,seconds=common.clock()-tick,
                truth_used_in_inference=False,new_acquisitions=0,rank=self.rank,fd=self.fd,
                center_limit=initial_center_limit,final_center_limit=center_limit,adaptive=adaptive,radius=radius,cycles=cycles)
        except (ValueError,RuntimeError) as error:
            return dict(command=-x,steps=history,actual_camera_calls=calls,network_calls=0,
                seconds=common.clock()-tick,stop_reason='model_error',error=str(error),truth_used_in_inference=False)
