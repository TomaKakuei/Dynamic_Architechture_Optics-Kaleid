"""2x2 line/pattern-search ablation inside exactly two original N updates."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
NROOT=HERE.parent/'olympus_secant_20261027'
KROOT=HERE.parent/'olympus_constrained_solver_20261026'
sys.path.insert(0,str(NROOT))
from secant import SecantRefiner,common,np,torch,Blocks,DiffMetrics,poses,sigma_from_adu
from qp import solve_local


class SearchAblation(SecantRefiner):
    def __init__(self):
        super().__init__()
        self.images_cached=[];self.calls=0;self.hits=0

    def begin_case(self,case):
        self.images_cached=[]
        path=HERE.parent/'olympus_weight_compare50_20261027/initial_images'/f'{case}.npz'
        if path.exists():
            with np.load(path) as z:self.images_cached.append((z['estimate'].copy(),z['image'].copy()))

    def render(self,q):
        for old,image in self.images_cached:
            if np.array_equal(old,q):
                self.hits+=1;return torch.as_tensor(image,device='cuda')
        self.calls+=1;image=self.camera.evaluate(q)[0]
        self.images_cached.append((q.copy(),image.cpu().numpy()))
        return image

    @torch.no_grad()
    def prepare(self,observed,prior,start,case,source_tag,last_cycle,center_limit):
        """Same cached initial 12D response as original N, before either update."""
        x=np.asarray(start,dtype=np.float64).copy();prior=np.asarray(prior,dtype=np.float64)
        obs=torch.as_tensor(observed,device='cuda').float();sigma=sigma_from_adu(obs)
        P=torch.as_tensor(self.precision,device='cuda').double()
        path=KROOT/'responses'/source_tag/f'{case}_{last_cycle}.npz'
        if not path.exists():path=KROOT/'responses/development_j_central8'/f'{case}_{last_cycle}.npz'
        with np.load(path) as z:
            q0=z['prior'].copy();basis=z['basis'].copy();B=basis[:,:8].copy()
            image0=torch.as_tensor(z['image'],device='cuda')
            raw=torch.stack([torch.as_tensor(z[f'jac_{i}'],device='cuda') for i in range(8)],dim=-1)
        coordinate=np.linalg.lstsq(B,x-q0,rcond=None)[0]
        if np.linalg.norm(B@coordinate-(x-q0))>1e-7:raise ValueError('K endpoint outside original response subspace')
        if np.array_equal(x,q0):current=image0
        else:
            with np.load(NROOT/'images'/f'{case}.npz') as z:
                if not np.array_equal(z['estimate'],x):raise ValueError('Original N initial-image mismatch')
                current=torch.as_tensor(z['image'],device='cuda')
        self.images_cached.append((x.copy(),current.cpu().numpy()))
        J=(raw/sigma[...,None]/128).flatten(0,-2).double()
        z=torch.as_tensor(coordinate,device='cuda').double();norm=float(z@z)
        if norm>=1e-16:
            # Match original N residual subtraction, including its arithmetic order.
            change=((current-obs)/sigma/128).flatten().double()-((image0-obs)/sigma/128).flatten().double()
            J=J+(change-J@z)[:,None]*z[None,:]/norm
        with np.load(NROOT/'extra_response'/f'{case}.npz') as z:
            if not np.array_equal(z['estimate'],x):raise ValueError('Original four extra-direction cache mismatch')
            columns=[(torch.as_tensor(z[f'jac_{i}'],device='cuda')/sigma/128).flatten().double() for i in range(8,12)]
        J=torch.cat([J,torch.stack(columns,dim=-1)],dim=1)
        return dict(x=x,prior=prior,obs=obs,sigma=sigma,P=P,B=basis[:,:12].copy(),J=J,
                    current=current,center_limit=center_limit)

    def bounded_pattern_coordinate(self,x,prior,B,coordinate,center_limit):
        """Shorten one ray from current x; do not project relative to the prior."""
        move=B@coordinate
        scale=min(1.,.05/max(float(abs(move).max()),1e-30))
        lo=np.maximum(-1.,prior-.2)-x;hi=np.minimum(1.,prior+.2)-x
        pos=move>1e-15;neg=move < -1e-15
        if pos.any():scale=min(scale,float((hi[pos]/move[pos]).min()))
        if neg.any():scale=min(scale,float((lo[neg]/move[neg]).min()))
        if center_limit:
            d=x-prior;G=self.center_gram;a=max(float(move@G@move),0.);b=2*float(d@G@move)
            c=float(d@G@d-center_limit**2)
            if c>1e-7:return None
            c=min(c,0.)
            if a>1e-20:scale=min(scale,max(0.,(-b+np.sqrt(max(b*b-4*a*c,0.)))/(2*a)))
            elif b>0:scale=min(scale,max(0.,-c/b))
        scale=max(0.,scale)
        if scale<1e-9 or np.max(abs(scale*move))<1e-9:return None
        return scale*coordinate

    @torch.no_grad()
    def predict_arm(self,prepared,arm):
        if arm not in ('N_line','N_pattern','N_line_pattern'):raise ValueError(arm)
        line=arm in ('N_line','N_line_pattern');pattern=arm in ('N_pattern','N_line_pattern')
        x=prepared['x'].copy();prior=prepared['prior'];obs=prepared['obs'];sigma=prepared['sigma'];P=prepared['P']
        B=prepared['B'];bt=torch.as_tensor(B,device='cuda').double();J=prepared['J'].clone()
        current=prepared['current'];limit=prepared['center_limit'];history=[];tick=common.clock()
        calls0=self.calls;hits0=self.hits;attempts=0
        def residual(im):return ((im-obs)/sigma/128).flatten().double()
        def merit(im,q):
            d=torch.as_tensor(q-prior,device='cuda').double()
            return float(residual(im).square().sum()+d@P@d)
        def finish(reason,**kw):
            return dict(command=-x,arm=arm,steps=history,seconds=common.clock()-tick,
                actual_camera_calls=self.calls-calls0,cache_hits=self.hits-hits0,proposal_attempts=attempts,
                stop_reason=reason,max_updates=2,truth_used_in_inference=False,new_acquisitions=0,**kw)
        try:
            cost=merit(current,x);initial=cost;floor=2*B.shape[1]/128**2
            for cycle in range(2):
                r=residual(current);d=torch.as_tensor(x-prior,device='cuda').double()
                Q=J.T@J+bt.T@P@bt;g=J.T@r+bt.T@P@d
                dz,info=solve_local(Q.cpu().numpy(),g.cpu().numpy(),B,x,prior,self.center_gram,.2,limit)
                before=cost;best=None;best_cost=cost;trials=[];line_values={0.:cost};axes=[]
                # All candidates in a cycle are based at this SAME x. There is
                # at most one accepted update and one secant update per cycle.
                def probe(coordinate,source,**metadata):
                    nonlocal best,best_cost,attempts
                    attempts+=1;trial=dict(source=source,**metadata)
                    if coordinate is None:
                        trial['constraint_rejected']=True;trials.append(trial);return None
                    u=torch.as_tensor(coordinate,device='cuda').double();q=x+B@coordinate
                    if not self.camera.camera.engine.geometry.path(q)['safe']:
                        trial['geometry_rejected']=True;trials.append(trial);return None
                    if not self.camera.camera.engine.geometry.path(prior-q,start=prior)['safe']:
                        trial['correction_path_rejected']=True;trials.append(trial);return None
                    try:image=self.render(q);value=merit(image,q)
                    except (ValueError,RuntimeError) as exc:
                        trial['error']=str(exc);trials.append(trial);return None
                    predicted=float((r+J@u).square().sum()+(d+bt@u)@P@(d+bt@u))
                    rho=(before-value)/max(before-predicted,1e-12)
                    acceptable=bool(value<before and rho>=.1)
                    trial.update(cost=value,predicted_cost=predicted,agreement=rho,acceptable=acceptable)
                    trials.append(trial)
                    if acceptable and value<best_cost:
                        best_cost=value;best=(q,image,u,source)
                    return trial
                if info['feasible'] and info['predicted_gain']>floor:
                    for gain in ((1.,.5,.25,.125,.0625) if line else (1.,.5,.25)):
                        trial=probe(gain*dz,'line_grid' if line else 'original_N_line',gain=gain)
                        if trial is not None and 'cost' in trial:line_values[gain]=trial['cost']
                        if not line and trial is not None and trial.get('acceptable'):break
                    if line and len(line_values)>=3:
                        points=sorted(line_values);j=min(range(len(points)),key=lambda i:line_values[points[i]])
                        left=max(0,min(j-1,len(points)-3));xs=np.asarray(points[left:left+3]);ys=np.asarray([line_values[a] for a in xs])
                        coeff=np.polyfit(xs,ys,2)
                        if coeff[0]>1e-12:
                            alpha=float(-coeff[1]/(2*coeff[0]))
                            if xs[0]<alpha<xs[-1] and min(abs(alpha-a) for a in points)>1e-6:
                                probe(alpha*dz,'line_quadratic',gain=alpha)
                    if pattern:
                        # Only original12 response coordinates. Observation/model
                        # gradient selects four axes; no case score or truth used.
                        ranking=(g.abs()/Q.diagonal().clamp_min(1e-12).sqrt()).cpu().numpy()
                        axes=np.argsort(-ranking,kind='stable')[:4].tolist()
                        for axis in axes:
                            width=.025/max(float(abs(B[:,axis]).max()),1e-12)
                            for sign in (-1.,1.):
                                coordinate=np.zeros(12);coordinate[axis]=sign*width
                                coordinate=self.bounded_pattern_coordinate(x,prior,B,coordinate,limit)
                                probe(coordinate,'direction_poll',axis=int(axis),sign=sign)
                accepted=best is not None;correction=None;selected=None
                if accepted:
                    q,image,u,selected=best;change=residual(image)-r;norm=float(u@u)
                    if norm>=1e-16:
                        error=change-J@u;correction=float(error.norm()/change.norm().clamp_min(1e-12))
                        J=J+error[:,None]*u[None,:]/norm
                    x=q;current=image;cost=best_cost
                history.append(dict(step=cycle+1,before=before,after=cost,accepted=accepted,solver=info,
                    trials=trials,pattern_axes=axes,selected_source=selected,secant_relative_correction=correction))
                if not accepted:break
            return finish('fixed_two_update_local_search',initial_image_cost=initial,final_image_cost=cost)
        except (ValueError,RuntimeError,OSError) as exc:
            return finish('model_error_retained_last_estimate',error=str(exc))
