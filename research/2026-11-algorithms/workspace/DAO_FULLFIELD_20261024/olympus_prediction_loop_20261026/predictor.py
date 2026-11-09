"""Frozen neural predictor loop and bounded local search on one fixed packet.

At estimate x, synthesize camera image F(x), predict g(F(x)) with the same
network, and propose x + [g(y)-g(F(x))]. Original y never changes. Actual
model-image likelihood selects bounded gains; no hidden state enters here.
"""
from context import *


class PredictorLoop(Refiner):
    def __init__(self):
        super().__init__('hybrid')
        self.network,_=restore(REPAIR/'training/last.pt')
        self.inv_cov=np.linalg.inv(self.cov)/128**2
        self.images=ROOT/'model_images';self.images.mkdir(exist_ok=True)

    @torch.no_grad()
    def predict_loop(self,observed,prior,start,key,cycles=2,radius=.1,center_limit=.01,noise_stop=8.,mode='neural'):
        observed=torch.as_tensor(observed,device='cuda').float();sigma=sigma_from_adu(observed)
        prior=np.asarray(prior,dtype=np.float64);x=np.array(start,copy=True);history=[];calls=0;nn_calls=0
        P=torch.as_tensor(self.inv_cov,device='cuda').double();tick=common.clock()
        def value(rendered,estimate):
            d=torch.as_tensor(estimate-prior,device='cuda').double()
            return float((((rendered-observed)/sigma/128).square().sum()))+float(d@P@d)
        def render(estimate):
            nonlocal calls
            calls+=1
            return self.camera.evaluate(estimate)[0]
        cached=self.images/f'initial_{key}.npz'
        try:
            if cached.exists():
                with np.load(cached) as z:
                    if not np.array_equal(z['estimate'],x):raise ValueError('Initial image cache belongs to different estimate')
                    current=torch.as_tensor(z['image'],device='cuda')
            elif np.array_equal(prior,x) and (self.cache_dir/f'{key}.npz').exists():
                with np.load(self.cache_dir/f'{key}.npz') as z:current=torch.as_tensor(z['image'].copy(),device='cuda')
            else:
                current=render(x);np.savez_compressed(cached,estimate=x,image=current.cpu().numpy())
        except (ValueError,RuntimeError) as error:
            return dict(command=-x,steps=[],actual_camera_calls=calls,network_calls=0,stop_reason='model_error',error=str(error),seconds=common.clock()-tick)
        cost=value(current,x);initial_cost=cost
        def project(proposal):
            move=proposal-prior
            scale=min(1.,radius/max(float(abs(move).max()),1e-30))
            center=float(np.sqrt(max(move@self.center_gram@move,0)))
            scale=min(scale,center_limit/max(center,1e-30)) if center_limit else scale
            move*=scale;proposal=prior+move
            if abs(proposal).max()>1:return None
            return proposal
        for cycle in range(cycles):
            if cost<=noise_stop:
                history.append(dict(cycle=cycle+1,action='observation_noise_stop',cost=cost));break
            # g(y) is the stored neural prior. g(F(x)) is a new forward of the
            # same frozen network on its known-model predicted camera image.
            simulated=-self.network.command(current[None])[0].double().cpu().numpy();nn_calls+=1
            direction=prior-simulated
            direction*=min(1.,.05/max(float(abs(direction).max()),1e-30))
            before=cost;best=None;trials=[]
            for gain in (1.,.5,-.5):
                proposal=project(x+gain*direction)
                if proposal is None or np.max(abs(proposal-x))<1e-8:continue
                try:
                    if not self.camera.camera.engine.geometry.path(proposal)['safe']:continue
                    if not self.camera.camera.engine.geometry.path(prior-proposal,start=prior)['safe']:continue
                    image=render(proposal);merit=value(image,proposal)
                except (ValueError,RuntimeError):continue
                trials.append(dict(source='neural_consistency',gain=gain,cost=merit))
                if np.isfinite(merit) and merit<cost:
                    cost=merit;best=(proposal,image)
            if mode=='pattern' and (best is None or (before-cost)/max(before,1e-12)<.01):
                # Cheap physical re-prediction of current sensitive directions;
                # only a small local pattern is rendered, not a full Jacobian.
                basis=self.hybrid_basis(x)
                for axis in (0,4):
                    for sign in (-1,1):
                        proposal=project(x+sign*.025*basis[:,axis])
                        if proposal is None:continue
                        try:
                            if not self.camera.camera.engine.geometry.path(proposal)['safe']:continue
                            if not self.camera.camera.engine.geometry.path(prior-proposal,start=prior)['safe']:continue
                            image=render(proposal);merit=value(image,proposal)
                        except (ValueError,RuntimeError):continue
                        trials.append(dict(source='local_pattern',axis=axis,sign=sign,cost=merit))
                        if np.isfinite(merit) and merit<cost:cost=merit;best=(proposal,image)
            if best is not None:x,current=best
            history.append(dict(cycle=cycle+1,before=before,after=cost,accepted=best is not None,trials=trials,
                max_departure=float(abs(x-prior).max()),predicted_center_change=float(np.sqrt(max((x-prior)@self.center_gram@(x-prior),0)))))
            if best is None:break
        return dict(command=-x,steps=history,actual_camera_calls=calls,network_calls=nn_calls,stop_reason='bounded_prediction_loop',
            initial_image_cost=initial_cost,final_image_cost=cost,seconds=common.clock()-tick,
            max_departure=float(abs(x-prior).max()),radius=radius,cycles=cycles,noise_stop=noise_stop,
            truth_used_in_inference=False,new_acquisitions=0)
