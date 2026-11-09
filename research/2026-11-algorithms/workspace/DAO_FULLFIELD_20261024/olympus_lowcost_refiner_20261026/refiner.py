"""Few complete-camera responses in train-calibrated optical directions."""
from bridge import *


class Refiner:
    def __init__(self,basis_mode='goal',resources=None):
        if resources is None:
            self.camera=FullCamera()
            with np.load(RUN/'calibration.npz') as z:
                self.basis=z['basis'].copy();self.cov=z['covariance'].copy()
            self.center_gram=np.load(RUN/'center_gram.npy') if (RUN/'center_gram.npy').exists() else None
        else:
            self.camera=resources.camera;self.basis=resources.basis;self.cov=resources.cov
            self.center_gram=resources.center_gram
        self.basis_mode=basis_mode
        self.cache_dir=RUN/('response_cache' if basis_mode=='goal' else 'response_hybrid');self.cache_dir.mkdir(exist_ok=True)

    @torch.no_grad()
    def hybrid_basis(self,prior):
        h=.001;plus=np.clip(prior+h*np.eye(35),-1,1);minus=np.clip(prior-h*np.eye(35),-1,1)
        qs=np.concatenate([prior[None],plus,minus]);phases=[];masks=[]
        for start in range(0,len(qs),8):
            phase,mask,_=self.camera.phase(torch.as_tensor(qs[start:start+8],device='cuda').double())
            phases.append(phase);masks.append(mask)
        phases=torch.cat(phases);masks=torch.cat(masks)
        width=np.diag(plus-minus).copy()
        mask=masks[0:1]&masks[1:36]&masks[36:]
        derivative=(phases[1:36]-phases[36:])/torch.as_tensor(width,device='cuda')[:,None,None]
        count=mask.sum(-1,keepdim=True).clamp_min(1)
        mean=torch.where(mask,derivative,0).sum(-1,keepdim=True)/count
        response=torch.where(mask,derivative-mean,0)/(9*count).sqrt()
        response=response.flatten(1);gram=(response@response.T).cpu().numpy()
        ev,U=np.linalg.eigh(self.cov);root=(U*np.sqrt(np.maximum(ev,1e-12)))@U.T
        _,V=np.linalg.eigh(root@gram@root);image_vectors=root@V[:,-4:][:,::-1]
        # Preserve current low-order/image-sensitive nuisance directions as
        # well as endpoint optical-risk directions; all use known model/prior.
        candidates=np.column_stack([image_vectors,self.basis]);chosen=[]
        for d in candidates.T:
            d=d.copy()
            for u in chosen:d-=u*(u@d)
            size=np.linalg.norm(d)
            if size>1e-8:chosen.append(d/size)
            if len(chosen)==35:break
        return np.column_stack(chosen)

    @torch.no_grad()
    def response(self,prior,key,rank):
        """Only packet ID labels the cache; it never chooses model behavior."""
        dest=self.cache_dir/f'{key}.npz';known={}
        if dest.exists():
            with np.load(dest) as z:known={k:z[k].copy() for k in z.files}
            if not np.array_equal(known['prior'],prior):raise ValueError('Cache belongs to a different neural prior')
        elif self.basis_mode!='goal' and (RUN/'response_cache'/f'{key}.npz').exists():
            with np.load(RUN/'response_cache'/f'{key}.npz') as z:
                if np.array_equal(z['prior'],prior):known=dict(image=z['image'].copy(),prior=prior)
        calls=0
        if 'image' not in known:
            prediction,_=self.camera.evaluate(prior);known['image']=prediction.cpu().numpy();known['prior']=prior
            np.savez_compressed(dest,**known);calls+=1
        prediction=torch.as_tensor(known['image'],device='cuda');columns=[];directions=[];hs=[]
        if rank and self.basis_mode=='hybrid' and 'hybrid_basis' not in known:
            known['hybrid_basis']=self.hybrid_basis(prior);np.savez_compressed(dest,**known)
        basis=known.get('hybrid_basis',self.basis)
        for col in range(rank):
            label=f'jac_{col}'
            if label not in known:
                base=basis[:,col]
                def allowed(d):
                    bound=np.where(d>0,(1-prior)/np.maximum(d,1e-30),(-1-prior)/np.minimum(d,-1e-30))
                    return min(.02,.9*float(bound[np.abs(d)>1e-12].min()))
                hp,hm=allowed(base),allowed(-base);direction=base if hp>=hm else -base;h=max(hp,hm)
                if h<1e-5:
                    direction=direction.copy();direction[(prior>=.99999)&(direction>0)]=0;direction[(prior<=-.99999)&(direction<0)]=0
                    h=allowed(direction) if np.linalg.norm(direction)>1e-10 else 0
                if h>1e-5:
                    probe,_=self.camera.evaluate(prior+h*direction);jac=(probe-prediction)/h;calls+=1
                else:jac=torch.zeros_like(prediction);direction=np.zeros(35)
                known[label]=jac.cpu().numpy();known[f'direction_{col}']=direction;known[f'h_{col}']=np.array(h)
                np.savez_compressed(dest,**known)
            columns.append(torch.as_tensor(known[label],device='cuda'));directions.append(known[f'direction_{col}']);hs.append(float(known[f'h_{col}']))
        return prediction,torch.stack(columns,dim=-1) if columns else None,np.stack(directions,axis=1) if directions else np.empty((35,0)),calls,hs

    @torch.no_grad()
    def predict(self,image,prior,key,rank=4,steps=1,radius=.1,step_radius=.05,prior_strength=1.,center_limit=0.):
        prior=np.asarray(prior,dtype=np.float64);x=prior.copy();history=[];start=common.clock()
        observed=torch.as_tensor(image,device='cuda').float();sigma=sigma_from_adu(observed)
        try:
            prediction,_,_,calls,_=self.response(prior,key,0)
            initial_cost=float((((prediction-observed)/sigma/128).square().sum()))
            if initial_cost<=9:
                return dict(command=-prior,history=[dict(step=0,action='noise_level_stop_before_derivatives',image_cost=initial_cost)],
                    stop_reason='noise_level',actual_camera_calls=calls,logical_camera_calls_without_cache=1,
                    active_seconds=common.clock()-start,rank=0,requested_rank=rank)
            prediction,j,B,extra_calls,hs=self.response(prior,key,rank);calls+=extra_calls
        except (ValueError,RuntimeError) as error:
            return dict(command=-prior,history=[],stop_reason='model_rejected_prior_retained',error=str(error),actual_camera_calls=0,
                        active_seconds=common.clock()-start,rank=rank)
        J=(j/sigma[...,None]/128).reshape(-1,rank).double()
        bt=torch.as_tensor(B,device='cuda').double()
        precision=B.T@np.linalg.solve(self.cov,B)/128**2*prior_strength
        P=torch.as_tensor(precision,device='cuda').double();G=J.T@J
        C=torch.as_tensor(B.T@self.center_gram@B,device='cuda').double() if center_limit else torch.zeros_like(G)
        z=torch.zeros(rank,device='cuda',dtype=torch.float64);identity=torch.eye(rank,device='cuda',dtype=torch.float64)
        nominal_calls=rank+1
        for step in range(steps):
            residual=((prediction-observed)/sigma/128).flatten().double();image_cost=float(residual.square().sum())
            if image_cost<=9:
                history.append(dict(step=step+1,action='noise_level_stop',image_cost=image_cost));break
            gradient=J.T@residual+P@z;damping=1e-6;center_mu=0.
            for _ in range(40):
                dz=-torch.linalg.solve(G+P+center_mu*C+damping*identity,gradient+center_mu*C@z);delta=bt@dz
                center_change=float(((z+dz)@C@(z+dz)).clamp_min(0).sqrt())
                center_good=not center_limit or center_change<=center_limit
                step_good=float(delta.abs().max())<=step_radius
                if center_good and step_good:break
                if not center_good:center_mu=10. if center_mu==0 else center_mu*4
                if not step_good:damping=max(1e-5,damping*4)
            before=image_cost+float(z@P@z);accepted=False;attempts=0
            for gain in (1.,.5):
                proposal=z+gain*dz;move=(bt@proposal).cpu().numpy()
                scale=min(1.,radius/max(float(abs(move).max()),1e-30));proposal=proposal*scale
                if center_limit:
                    predicted_center=float((proposal@C@proposal).clamp_min(0).sqrt())
                    proposal=proposal*min(1.,center_limit/max(predicted_center,1e-30))
                trial=prior+(bt@proposal).cpu().numpy();attempts+=1
                if abs(trial).max()>1:continue
                try:
                    if not self.camera.camera.engine.geometry.path(trial)['safe']:continue
                    if not self.camera.camera.engine.geometry.path(prior-trial,start=prior)['safe']:continue
                    rendered,_=self.camera.evaluate(trial);calls+=1;nominal_calls+=1
                    cost=float((((rendered-observed)/sigma/128).square().sum()))+float(proposal@P@proposal)
                except (ValueError,RuntimeError):continue
                if np.isfinite(cost) and cost<before:
                    prediction=rendered;x=trial;z=proposal;accepted=True;break
            history.append(dict(step=step+1,before=before,after=cost if accepted else before,
                accepted=accepted,damping=damping,center_mu=center_mu,predicted_center_change=float((z@C@z).clamp_min(0).sqrt()),
                attempts=attempts,max_departure=float(abs(x-prior).max())))
            if not accepted:break
        return dict(command=-x,history=history,stop_reason='bounded_budget',actual_camera_calls=calls,
                    logical_camera_calls_without_cache=nominal_calls,active_seconds=common.clock()-start,rank=rank,
                    complete_response_in_subspace=True,total_prior_departure_limit=radius,prior_strength=prior_strength,center_limit=center_limit)
