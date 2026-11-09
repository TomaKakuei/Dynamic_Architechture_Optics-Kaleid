"""Re-predict and relinearize on the same measured packet, with local search."""
from context import *
from predictor import PredictorLoop


class ModelLoop(PredictorLoop):
    def __init__(self):
        super().__init__()
        self.cache_dir=ROOT/'loop_responses';self.cache_dir.mkdir(exist_ok=True)

    @torch.no_grad()
    def predict_loop(self,observed,prior,start,key,cycles=2,radius=.2,center_limit=.01,noise_stop=9.,mode='relinearized'):
        observed=torch.as_tensor(observed,device='cuda').float();sigma=sigma_from_adu(observed)
        prior=np.asarray(prior,dtype=np.float64);x=np.asarray(start,dtype=np.float64).copy()
        history=[];calls=0;tick=common.clock();Pfull=torch.as_tensor(self.inv_cov,device='cuda').double()
        def render(q):
            nonlocal calls
            calls+=1;return self.camera.evaluate(q)[0]
        def merit(image,q):
            d=torch.as_tensor(q-prior,device='cuda').double()
            return float((((image-observed)/sigma/128).square().sum()))+float(d@Pfull@d)
        def feasible(q):
            d=q-prior;scale=min(1.,radius/max(float(abs(d).max()),1e-30))
            center=float(np.sqrt(max(d@self.center_gram@d,0)))
            scale=min(scale,center_limit/max(center,1e-30)) if center_limit else scale
            q=prior+d*scale
            if abs(q).max()>1:return None
            if not self.camera.camera.engine.geometry.path(q)['safe']:return None
            if not self.camera.camera.engine.geometry.path(prior-q,start=prior)['safe']:return None
            return q
        cached=self.images/f'initial_{key}.npz'
        try:
            if cached.exists():
                with np.load(cached) as z:
                    if not np.array_equal(z['estimate'],x):raise ValueError('Initial estimate cache mismatch')
                    current=torch.as_tensor(z['image'],device='cuda')
            else:current=render(x);np.savez_compressed(cached,estimate=x,image=current.cpu().numpy())
        except (ValueError,RuntimeError) as exc:
            return dict(command=-x,steps=[],actual_camera_calls=calls,network_calls=0,stop_reason='model_error',error=str(exc),seconds=common.clock()-tick)
        cost=merit(current,x);initial=cost;centerG=torch.as_tensor(self.center_gram,device='cuda').double()
        for cycle in range(cycles):
            if cost<=noise_stop:history.append(dict(cycle=cycle+1,action='noise_stop',cost=cost));break
            response_key=f'{key}_{cycle}_{radius}_{center_limit}'
            response_path=self.cache_dir/f'{response_key}.npz'
            if not response_path.exists():np.savez_compressed(response_path,prior=x,image=current.cpu().numpy())
            try:
                prediction,j,B,ncalls,_=self.response(x,response_key,8);calls+=ncalls
            except (ValueError,RuntimeError) as error:
                history.append(dict(cycle=cycle+1,action='response_failed',error=str(error)));break
            J=(j/sigma[...,None]/128).reshape(-1,8).double();bt=torch.as_tensor(B,device='cuda').double()
            d=torch.as_tensor(x-prior,device='cuda').double();r=((prediction-observed)/sigma/128).flatten().double()
            G=J.T@J;P=bt.T@Pfull@bt;C=bt.T@centerG@bt
            gradient=J.T@r+bt.T@Pfull@d;mu=0.;damping=1e-6;I=torch.eye(8,device='cuda',dtype=torch.float64)
            for _ in range(45):
                dz=-torch.linalg.solve(G+P+mu*C+damping*I,gradient+mu*bt.T@centerG@d)
                delta=bt@dz;newd=d+delta
                center_good=not center_limit or float((newd@centerG@newd).clamp_min(0).sqrt())<=center_limit
                step_good=float(delta.abs().max())<=.05
                if center_good and step_good:break
                if not center_good:mu=10. if mu==0 else mu*4
                if not step_good:damping=max(1e-5,damping*4)
            direction=delta.cpu().numpy();before=cost;best=None;trials=[]
            for gain in (1.,.5,1.5):
                try:
                    q=feasible(x+gain*direction)
                    if q is None:continue
                    im=render(q);v=merit(im,q)
                except (ValueError,RuntimeError):continue
                trials.append(dict(source='fresh_linearization',gain=gain,cost=v))
                if np.isfinite(v) and v<cost:cost=v;best=(q,im)
            if best is None or (before-cost)/max(before,1e-12)<.02:
                # Search extra directions outside the previously retained8.
                with np.load(response_path) as z:basis=z['hybrid_basis'].copy()
                for axis in (8,9):
                    for sign in (-1,1):
                        try:
                            q=feasible(x+sign*.025*basis[:,axis])
                            if q is None:continue
                            im=render(q);v=merit(im,q)
                        except (ValueError,RuntimeError):continue
                        trials.append(dict(source='local_extra_direction',axis=axis,sign=sign,cost=v))
                        if np.isfinite(v) and v<cost:cost=v;best=(q,im)
            if best is not None:x,current=best
            history.append(dict(cycle=cycle+1,before=before,after=cost,accepted=best is not None,trials=trials,
                damping=damping,center_mu=mu,max_departure=float(abs(x-prior).max())))
            if best is None:break
        return dict(command=-x,steps=history,actual_camera_calls=calls,network_calls=0,stop_reason='bounded_model_loop',
            initial_image_cost=initial,final_image_cost=cost,seconds=common.clock()-tick,max_departure=float(abs(x-prior).max()),
            radius=radius,cycles=cycles,truth_used_in_inference=False,new_acquisitions=0,
            relinearized_each_cycle=True,local_extra_direction_search=True)
