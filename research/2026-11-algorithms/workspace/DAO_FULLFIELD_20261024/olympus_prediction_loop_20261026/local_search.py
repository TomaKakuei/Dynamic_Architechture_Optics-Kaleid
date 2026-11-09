"""Two-direction quadratic local search after the fixed model loop.

Accept only an actual same-packet image-merit decrease. No truth input.
The nominal center trust ellipsoid and original pose/path checks are retained.
"""
from context import *
from predictor import PredictorLoop


class QuadraticLocalSearch(PredictorLoop):
    @torch.no_grad()
    def predict_local(self, observed, prior, start, key, radius=.2, center_limit=.01, noise_stop=9.):
        observed=torch.as_tensor(observed,device='cuda').float()
        sigma=sigma_from_adu(observed)
        prior=np.asarray(prior,dtype=np.float64)
        x=np.asarray(start,dtype=np.float64).copy()
        P=torch.as_tensor(self.inv_cov,device='cuda').double()
        calls=0; history=[]; tick=common.clock()
        def render(q):
            nonlocal calls
            calls+=1
            return self.camera.evaluate(q)[0]
        def merit(im,q):
            d=torch.as_tensor(q-prior,device='cuda').double()
            return float((((im-observed)/sigma/128).square().sum()))+float(d@P@d)
        def feasible(q):
            d=q-prior
            center=float(np.sqrt(max(d@self.center_gram@d,0)))
            scale=min(1.,radius/max(float(abs(d).max()),1e-30),center_limit/max(center,1e-30))
            q=prior+scale*d
            if abs(q).max()>1:return None
            if not self.camera.camera.engine.geometry.path(q)['safe']:return None
            if not self.camera.camera.engine.geometry.path(prior-q,start=prior)['safe']:return None
            return q
        try:
            cache=self.images/f'local_start_{key}.npz'
            if cache.exists():
                with np.load(cache) as z:
                    if not np.array_equal(z['estimate'],x):raise ValueError('Local start cache mismatch')
                    current=torch.as_tensor(z['image'],device='cuda')
            else:
                current=render(x)
                np.savez_compressed(cache,estimate=x,image=current.cpu().numpy())
            cost=merit(current,x); initial=cost
            if cost>noise_stop:
                basis=self.hybrid_basis(x)
                for axis in (8,9):
                    if cost<=noise_stop:break
                    base=x.copy(); before=cost; direction=basis[:,axis].copy()
                    d=x-prior; center=np.sqrt(max(d@self.center_gram@d,0))
                    tangent=bool(center>=.9*center_limit)
                    if tangent:
                        normal=self.center_gram@d
                        direction-=normal*(normal@direction)/max(normal@normal,1e-30)
                    length=np.linalg.norm(direction)
                    if length<1e-8:continue
                    direction/=length
                    width=.025; values={}; trials=[]; best=None
                    def probe(step,source):
                        nonlocal best,cost
                        q=feasible(base+step*direction)
                        if q is None:return None
                        try:im=render(q); value=merit(im,q)
                        except (ValueError,RuntimeError):return None
                        trials.append(dict(source=source,step=step,cost=value))
                        if np.isfinite(value) and value<cost:cost=value;best=(q,im)
                        return value
                    for sign in (-1,1):values[sign]=probe(sign*width,'bracket')
                    if all(v is not None and np.isfinite(v) for v in values.values()):
                        denom=values[1]+values[-1]-2*before
                        if denom>1e-10:
                            step=float(np.clip(.5*width*(values[-1]-values[1])/denom,-width,width))
                            if 1e-6<abs(step)<.99*width:probe(step,'quadratic_interpolation')
                    if best is not None:x,current=best
                    history.append(dict(axis=axis,tangent_to_active_center=tangent,before=before,
                                        after=cost,accepted=best is not None,trials=trials))
            return dict(command=-x,steps=history,actual_camera_calls=calls,network_calls=0,
                initial_image_cost=initial,final_image_cost=cost,seconds=common.clock()-tick,
                stop_reason='bounded_quadratic_search',max_departure=float(abs(x-prior).max()),
                truth_used_in_inference=False,new_acquisitions=0)
        except (ValueError,RuntimeError) as error:
            return dict(command=-x,steps=history,actual_camera_calls=calls,network_calls=0,
                seconds=common.clock()-tick,stop_reason='local_model_error',error=str(error),
                truth_used_in_inference=False,new_acquisitions=0)
