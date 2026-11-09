"""One configuration: two N updates, up to64 objective queries per update."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'olympus_local_ablation150_20261027'
sys.path.insert(0,str(PREVIOUS))
from search import SearchAblation,common,np,torch,Blocks,DiffMetrics,poses,solve_local,KROOT
from scipy.optimize import minimize,LinearConstraint,NonlinearConstraint


class SearchBudgetReached(Exception):pass


class WideSearch(SearchAblation):
    @torch.no_grad()
    def predict(self,prepared,progress=None):
        x=prepared['x'].copy();prior=prepared['prior'];obs=prepared['obs'];sigma=prepared['sigma'];P=prepared['P']
        B=prepared['B'];bt=torch.as_tensor(B,device='cuda').double();J=prepared['J'].clone()
        current=prepared['current'];limit=prepared['center_limit'];history=[];tick=common.clock()
        calls0=self.calls;hits0=self.hits;attempts=0
        def residual(im):return ((im-obs)/sigma/128).flatten().double()
        def merit(im,q):
            d=torch.as_tensor(q-prior,device='cuda').double()
            return float(residual(im).square().sum()+d@P@d)
        def finish(reason,**kw):
            return dict(command=-x,steps=history,seconds=common.clock()-tick,
                actual_camera_calls=self.calls-calls0,cache_hits=self.hits-hits0,objective_requests=attempts,
                stop_reason=reason,max_updates=2,max_objective_requests_per_loop=64,
                truth_used_in_inference=False,new_acquisitions=0,**kw)
        try:
            cost=merit(current,x);initial=cost;floor=2*B.shape[1]/128**2
            for cycle in range(2):
                r=residual(current);d=torch.as_tensor(x-prior,device='cuda').double()
                Q=J.T@J+bt.T@P@bt;g=J.T@r+bt.T@P@d
                dz,info=solve_local(Q.cpu().numpy(),g.cpu().numpy(),B,x,prior,self.center_gram,.2,limit)
                before=cost;best=None;best_cost=cost;best_coordinate=np.zeros(12)
                trials=[];loop_requests=0;optimizer=None;start_calls=self.calls
                delta0=x-prior
                lo=np.maximum.reduce([np.full(35,-.05),-.2-delta0,-1-x])
                hi=np.minimum.reduce([np.full(35,.05),.2-delta0,1-x])
                variable_scale=.05;A=variable_scale*B

                def evaluate(coordinate,source,**metadata):
                    nonlocal best,best_cost,best_coordinate,loop_requests,attempts
                    if loop_requests>=64:raise SearchBudgetReached()
                    loop_requests+=1;attempts+=1
                    coordinate=np.asarray(coordinate,dtype=float)
                    move=B@coordinate;q=x+move
                    center=float(np.sqrt(max((delta0+move)@self.center_gram@(delta0+move),0)))
                    violation=max(float(np.max(lo-move)),float(np.max(move-hi)),0.)
                    feasible=bool(violation<=1e-7 and (not limit or center<=limit*(1+1e-5)))
                    trial=dict(source=source,coordinate=coordinate.tolist(),feasible=feasible,
                        max_box_violation=violation,nominal_center_change=center,**metadata)
                    if progress is not None and (loop_requests==1 or loop_requests%8==0):
                        progress(cycle+1,loop_requests,best_cost)
                    # Trust-region-infeasible virtual probes may help fit the
                    # quadratic search model. Only feasible points can be accepted.
                    # Physical domain and geometry remain hard for every rendering.
                    if float(abs(q).max())>1+1e-7:
                        trial['physical_domain_rejected']=True;trials.append(trial);return before+1e6*(1+violation),False
                    if not self.camera.camera.engine.geometry.path(q)['safe']:
                        trial['geometry_rejected']=True;trials.append(trial);return before+1e6,False
                    if not self.camera.camera.engine.geometry.path(prior-q,start=prior)['safe']:
                        trial['correction_path_rejected']=True;trials.append(trial);return before+1e6,False
                    try:
                        image=self.render(q);value=merit(image,q)
                    except (ValueError,RuntimeError) as exc:
                        trial['error']=str(exc);trials.append(trial);return before+1e6,False
                    if not np.isfinite(value):
                        trial['nonfinite_cost']=True;trials.append(trial);return before+1e6,False
                    u=torch.as_tensor(coordinate,device='cuda').double()
                    predicted=float((r+J@u).square().sum()+(d+bt@u)@P@(d+bt@u))
                    rho=(before-value)/max(before-predicted,1e-12)
                    acceptable=bool(feasible and value<before and rho>=.1)
                    trial.update(cost=value,predicted_cost=predicted,agreement=rho,acceptable=acceptable)
                    trials.append(trial)
                    if acceptable and value<best_cost:
                        best_cost=value;best=(q,image,u,source);best_coordinate=coordinate.copy()
                    return value,acceptable

                if info['feasible'] and info['predicted_gain']>floor:
                    # Original N seed and all later search evaluations share the
                    # same64-query cap. No hidden extra local-search loops.
                    for gain in (1.,.5,.25):
                        _,accepted=evaluate(gain*dz,'original_N_seed',gain=gain)
                        if accepted:break
                    constraints=[LinearConstraint(A,lo,hi)]
                    if limit:
                        def center_constraint(z):
                            v=delta0+A@z
                            return float(v@self.center_gram@v/limit**2)
                        constraints.append(NonlinearConstraint(center_constraint,-np.inf,1.))
                    z0=best_coordinate/variable_scale
                    remaining=64-loop_requests
                    try:
                        solved=minimize(lambda z:evaluate(variable_scale*z,'joint_COBYQA')[0],z0,
                            method='COBYQA',constraints=constraints,
                            options=dict(maxfev=remaining,maxiter=1000,initial_tr_radius=.25,
                                         final_tr_radius=.01,feasibility_tol=1e-8,scale=False,disp=False))
                        optimizer=dict(success=bool(solved.success),status=int(solved.status),message=str(solved.message),
                            nfev=int(solved.nfev),nit=int(solved.nit),fun=float(solved.fun),
                            maxcv=float(getattr(solved,'maxcv',0.)),budget_limited=loop_requests>=64)
                    except SearchBudgetReached:
                        optimizer=dict(success=False,message='Shared64-evaluation cap reached',budget_limited=True)
                accepted=best is not None;correction=None;selected=None
                if accepted:
                    q,image,u,selected=best;change=residual(image)-r;norm=float(u@u)
                    if norm>=1e-16:
                        error=change-J@u;correction=float(error.norm()/change.norm().clamp_min(1e-12))
                        J=J+error[:,None]*u[None,:]/norm
                    # The sole outer state/response update in this cycle.
                    x=q;current=image;cost=best_cost
                history.append(dict(step=cycle+1,before=before,after=cost,accepted=accepted,solver=info,
                    search=optimizer,trials=trials,objective_requests=loop_requests,
                    actual_camera_calls=self.calls-start_calls,selected_source=selected,
                    secant_relative_correction=correction))
                if progress is not None:progress(cycle+1,loop_requests,cost)
                if not accepted:break
            return finish('two_N_loops_joint_search64',initial_image_cost=initial,final_image_cost=cost)
        except (ValueError,RuntimeError,OSError) as exc:
            return finish('model_error_retained_last_estimate',error=str(exc))
