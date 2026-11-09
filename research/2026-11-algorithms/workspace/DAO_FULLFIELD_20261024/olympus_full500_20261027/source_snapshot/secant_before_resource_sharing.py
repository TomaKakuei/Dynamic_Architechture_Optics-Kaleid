"""Two low-cost model-only continuation steps using existing full-camera responses."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'olympus_constrained_solver_20261026'
sys.path.insert(0,str(BASE))
from solver import Refiner,common,np,torch,Blocks,DiffMetrics,baseline,poses,sigma_from_adu,PREVIOUS
from qp import solve_local


class SecantRefiner(Refiner):
    def __init__(self):
        super().__init__('hybrid')
        self.precision=np.linalg.inv(self.cov)/128**2
        self.images=ROOT/'images';self.images.mkdir(exist_ok=True)

    @torch.no_grad()
    def predict(self,observed,prior,start,key,source_tag,last_cycle,center_limit,steps=2,extra=0):
        x=np.asarray(start,dtype=np.float64).copy();prior=np.asarray(prior,dtype=np.float64)
        tick=common.clock();history=[];calls=0;fd_calls=0
        if last_cycle is None:
            return dict(command=-x,steps=[],seconds=common.clock()-tick,actual_camera_calls=0,stop_reason='no_active_K_response')
        obs=torch.as_tensor(observed,device='cuda').float();sigma=sigma_from_adu(obs)
        P=torch.as_tensor(self.precision,device='cuda').double()
        def render(q):
            nonlocal calls
            calls+=1
            return self.camera.evaluate(q)[0]
        def residual(im):return ((im-obs)/sigma/128).flatten().double()
        def merit(im,q):
            d=torch.as_tensor(q-prior,device='cuda').double()
            return float(residual(im).square().sum()+d@P@d)
        def update(J,change,coordinate):
            norm=float(coordinate@coordinate)
            if norm<1e-16:return J,0.
            error=change-J@coordinate
            relative=float(error.norm()/change.norm().clamp_min(1e-12))
            return J+error[:,None]*coordinate[None,:]/norm,relative
        try:
            cache=BASE/'responses'/source_tag/f'{key}_{last_cycle}.npz'
            if not cache.exists():cache=BASE/'responses'/'development_j_central8'/f'{key}_{last_cycle}.npz'
            with np.load(cache) as z:
                q0=z['prior'].copy();all_basis=z['basis'].copy();B=all_basis[:,:8].copy()
                image0=torch.as_tensor(z['image'],device='cuda')
                raw=torch.stack([torch.as_tensor(z[f'jac_{k}'],device='cuda') for k in range(8)],dim=-1)
            bt=torch.as_tensor(B,device='cuda').double()
            coordinate0=np.linalg.lstsq(B,x-q0,rcond=None)[0]
            if np.linalg.norm(B@coordinate0-(x-q0))>1e-7:raise ValueError('Last K move outside saved response subspace')
            image_cache=self.images/f'{key}.npz'
            if np.array_equal(x,q0):current=image0
            elif image_cache.exists():
                with np.load(image_cache) as z:
                    if not np.array_equal(z['estimate'],x):raise ValueError('Continuation image cache mismatch')
                    current=torch.as_tensor(z['image'],device='cuda')
            else:
                current=render(x);np.savez_compressed(image_cache,estimate=x,image=current.cpu().numpy())
            J=(raw/sigma[...,None]/128).flatten(0,-2).double()
            J,initial_secant_error=update(J,residual(current)-residual(image0),torch.as_tensor(coordinate0,device='cuda').double())
            if extra:
                extra_cache=ROOT/'extra_response';extra_cache.mkdir(exist_ok=True)
                dest=extra_cache/f'{key}.npz';known={}
                if dest.exists():
                    with np.load(dest) as z:
                        if not np.array_equal(z['estimate'],x):raise ValueError('Extra-direction cache estimate mismatch')
                        known={k:z[k].copy() for k in z.files}
                else:known=dict(estimate=x,basis=all_basis)
                columns=[]
                for axis in range(8,8+extra):
                    label=f'jac_{axis}';direction=all_basis[:,axis]
                    if label not in known:
                        mask=np.abs(direction)>1e-12
                        h=min(.005,.9*float(((1-np.abs(x[mask]))/np.abs(direction[mask])).min()))
                        if h<1e-6:raise ValueError('Insufficient symmetric probe space')
                        plus=render(x+h*direction);minus=render(x-h*direction);fd_calls+=2
                        known[label]=((plus-minus)/(2*h)).cpu().numpy();known[f'h_{axis}']=h
                        np.savez_compressed(dest,**known)
                    columns.append((torch.as_tensor(known[label],device='cuda')/sigma/128).flatten().double())
                J=torch.cat([J,torch.stack(columns,dim=-1)],dim=1)
                B=all_basis[:,:8+extra];bt=torch.as_tensor(B,device='cuda').double()
            cost=merit(current,x);initial=cost
            # Heuristic directional noise scale, not a calibrated significance test.
            gain_floor=2*B.shape[1]/128**2
            for cycle in range(steps):
                r=residual(current);d=torch.as_tensor(x-prior,device='cuda').double()
                Q=J.T@J+bt.T@P@bt;g=J.T@r+bt.T@P@d
                dz,info=solve_local(Q.cpu().numpy(),g.cpu().numpy(),B,x,prior,self.center_gram,.2,center_limit)
                before=cost;accepted=False;trials=[];secant_error=None
                if info['feasible'] and info['predicted_gain']>gain_floor:
                    for gain in (1.,.5,.25):
                        u=torch.as_tensor(gain*dz,device='cuda').double();q=x+B@(gain*dz)
                        if not self.camera.camera.engine.geometry.path(q)['safe']:continue
                        if not self.camera.camera.engine.geometry.path(prior-q,start=prior)['safe']:continue
                        try:im=render(q);value=merit(im,q)
                        except (ValueError,RuntimeError) as exc:
                            trials.append(dict(gain=gain,error=str(exc)));continue
                        predicted=float((r+J@u).square().sum()+(d+bt@u)@P@(d+bt@u))
                        rho=(before-value)/max(before-predicted,1e-12)
                        trials.append(dict(gain=gain,cost=value,predicted_cost=predicted,agreement=rho))
                        if value<before and rho>=.1:
                            J,secant_error=update(J,residual(im)-r,u)
                            x=q;current=im;cost=value;accepted=True;break
                history.append(dict(step=cycle+1,before=before,after=cost,accepted=accepted,solver=info,
                    trials=trials,secant_relative_correction=secant_error))
                if not accepted:break
            return dict(command=-x,steps=history,seconds=common.clock()-tick,actual_camera_calls=calls,
                initial_image_cost=initial,final_image_cost=cost,initial_secant_relative_correction=initial_secant_error,
                center_limit=center_limit,stop_reason='bounded_secant_continuation',new_acquisitions=0,
                truth_used_in_inference=False,new_finite_difference_probes=fd_calls,extra_directions=extra)
        except (ValueError,RuntimeError,OSError) as exc:
            return dict(command=-x,steps=history,seconds=common.clock()-tick,actual_camera_calls=calls,
                stop_reason='model_error_retained_last_estimate',error=str(exc),truth_used_in_inference=False)
