"""Paper beta_g proposal gains, not optical-information weights.

All routes start from the exact saved N estimate and reuse one observed packet.
The frozen predictor proposes g(y)-g(F(x)); only the group gains differ.
"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'olympus_prediction_loop_20261026'
sys.path.insert(0,str(OLD))
from context import Refiner,REPAIR,restore,common,np,torch,sigma_from_adu,Blocks,DiffMetrics,poses


class GroupLoop(Refiner):
    def __init__(self):
        super().__init__('hybrid')
        self.network,_=restore(REPAIR/'training/last.pt')
        self.precision=np.linalg.inv(self.cov)/128**2
        self.cache=[];self.network_cache=[];self.camera_calls=0;self.cache_hits=0;self.network_calls=0
        self.initials=HERE/'initial_images';self.initials.mkdir(exist_ok=True)

    def begin_case(self,case,start):
        self.cache=[];self.network_cache=[]
        self.case=case;self.initial_estimate=np.asarray(start).copy()
        path=self.initials/f'{case}.npz'
        if path.exists():
            with np.load(path) as z:
                if not np.array_equal(z['estimate'],start):raise ValueError('Initial cache mismatch')
                self.cache.append((z['estimate'].copy(),z['image'].copy()))

    def render(self,q):
        for old,image in self.cache:
            if np.array_equal(q,old):
                self.cache_hits+=1
                return torch.as_tensor(image,device='cuda')
        self.camera_calls+=1
        image=self.camera.evaluate(q)[0]
        self.cache.append((q.copy(),image.cpu().numpy()))
        if np.array_equal(q,self.initial_estimate):
            np.savez_compressed(self.initials/f'{self.case}.npz',estimate=q,image=image.cpu().numpy())
        return image

    def network_estimate(self,q,image):
        for old,value in self.network_cache:
            if np.array_equal(q,old):return value.copy()
        self.network_calls+=1
        value=-self.network.command(image[None])[0].double().cpu().numpy()
        self.network_cache.append((q.copy(),value.copy()))
        return value

    def feasible_move(self,x,prior,move,center_limit):
        """One common scale preserves the relative seven group gains."""
        scale=min(1.,.05/max(float(abs(move).max()),1e-30))
        low=np.maximum(-1.,prior-.2)-x; high=np.minimum(1.,prior+.2)-x
        pos=move>1e-15;neg=move < -1e-15
        if pos.any():scale=min(scale,float((high[pos]/move[pos]).min()))
        if neg.any():scale=min(scale,float((low[neg]/move[neg]).min()))
        if center_limit:
            d=x-prior;G=self.center_gram
            a=max(float(move@G@move),0.);b=2*float(d@G@move)
            c=float(d@G@d-center_limit**2)
            if c>1e-7:return None,0.
            c=min(c,0.)
            if a>1e-20:
                scale=min(scale,max(0.,(-b+np.sqrt(max(b*b-4*a*c,0.)))/(2*a)))
            elif b>0:scale=min(scale,max(0.,-c/b))
        scale=max(0.,scale)
        if scale<=1e-10:return None,scale
        q=x+scale*move
        if np.max(abs(q-x))<1e-9:return None,scale
        if not self.camera.camera.engine.geometry.path(q)['safe']:return None,scale
        if not self.camera.camera.engine.geometry.path(prior-q,start=prior)['safe']:return None,scale
        return q,scale

    @torch.no_grad()
    def predict(self,observed,prior,start,center_limit,route,cycles=4):
        if route not in ('equal_loop','equal_search','unequal_search'):raise ValueError(route)
        tick=common.clock();calls0=self.camera_calls;hits0=self.cache_hits;nn0=self.network_calls
        x=np.asarray(start,dtype=float).copy();prior=np.asarray(prior,dtype=float)
        obs=torch.as_tensor(observed,device='cuda').float();sigma=sigma_from_adu(obs)
        P=torch.as_tensor(self.precision,device='cuda').double();history=[];attempts=0
        def finish(reason,**kw):
            return dict(command=-x,route=route,steps=history,seconds=common.clock()-tick,
                actual_camera_calls=self.camera_calls-calls0,cache_hits=self.cache_hits-hits0,
                network_calls=self.network_calls-nn0,proposal_trials=attempts,
                stop_reason=reason,truth_used_in_inference=False,new_acquisitions=0,**kw)
        def merit(im,q):
            d=torch.as_tensor(q-prior,device='cuda').double()
            return float((((im-obs)/sigma/128).square().sum())+d@P@d)
        try:
            current=self.render(x);cost=merit(current,x);initial=cost
            for cycle in range(cycles):
                prediction=self.network_estimate(x,current)
                direction=prior-prediction
                direction*=min(1.,.025/max(float(abs(direction).max()),1e-30))
                if np.linalg.norm(direction)<1e-8:break
                before=cost;trials=[];best=None;best_beta=np.zeros(7)
                # Both search routes get exactly16 proposal slots per cycle;
                # zero motion is always available without a fresh rendering.
                if route=='equal_loop':schedule=[('common',1.)]
                elif route=='equal_search':
                    schedule=[('common',a) for a in (1.,-1.,.5,-.5,2.,-2.,.25,-.25,1.5,-1.5,.75,-.75,.125,-.125,.0625,-.0625)]
                else:
                    schedule=[('common',1.),('common',-1.)]
                    # Poll each group's gain once in both directions. The best
                    # accepted beta is the center for subsequent coordinate polls.
                    for group in range(7):schedule.extend([('group',(group,1.)),('group',(group,-1.))])
                for kind,value in schedule:
                    attempts+=1
                    if kind=='common':beta=np.full(7,value)
                    else:
                        group,sign=value;beta=best_beta.copy();beta[group]=np.clip(beta[group]+sign,-2,2)
                    move=(direction.reshape(7,5)*beta[:,None]).reshape(35)
                    q,scale=self.feasible_move(x,prior,move,center_limit)
                    trial=dict(beta=beta.tolist(),common_feasibility_scale=scale)
                    if q is None:
                        trial['constraint_rejected']=True;trials.append(trial);continue
                    try:
                        image=self.render(q);candidate_cost=merit(image,q)
                    except (ValueError,RuntimeError) as exc:
                        trial['error']=str(exc);trials.append(trial);continue
                    trial['cost']=candidate_cost;trials.append(trial)
                    if np.isfinite(candidate_cost) and candidate_cost<cost-1e-10:
                        cost=candidate_cost;best=(q,image);best_beta=beta.copy()
                if best is not None:x,current=best
                history.append(dict(cycle=cycle+1,before=before,after=cost,accepted=best is not None,
                    beta=best_beta.tolist(),trials=trials,
                    nominal_center_change=float(np.sqrt(max((x-prior)@self.center_gram@(x-prior),0)))))
                if best is None:break
            return finish('bounded_group_gain_loop',initial_image_cost=initial,final_image_cost=cost)
        except (ValueError,RuntimeError,OSError) as exc:
            return finish('model_error_retained_last_estimate',error=str(exc))
