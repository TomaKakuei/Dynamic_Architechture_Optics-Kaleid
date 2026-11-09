"""Round2: factorial noise/linearization diagnostic, fixed4steps and same24cases."""
from nikon_lab import *
import json
lab=Lab();start=active_time();q=lab.q0.copy();n=len(q)
jn=lab.jacobian(np.zeros((1,35)),'native')[0]
ja=lab.jacobian(np.zeros((1,35)),'audit')[0]
pinv=inverse(jn);refa=lab.observation(np.zeros((1,35)),'audit')[0]
u,s,v=np.linalg.svd(jn,full_matrices=False)
final=[]
for seed,h in zip(lab.seeds,lab.hidden):
    saved=json.loads((HERE.parent/'nikon_debug_20261024/validation24_cases'/f'{seed}_diff_calibrated.json').read_text())
    final.append(h+np.asarray(saved['command']))
final=np.asarray(final)
decomposition={};arrays={}
for stage,states in [('oneshot',q),('previous_closed_loop',final)]:
    observed=lab.observation(states)-lab.ref
    actual=lab.observation(states,'audit')-refa
    ideal_linear=states@ja.T
    inverse_linear=(states@jn.T)@pinv.T@ja.T
    inverse_nonlinear=observed@pinv.T@ja.T
    noise=lab.noise(lab.seeds,0)@pinv.T@ja.T
    def error(x):
        # Audit vector is field residual/sqrt(pupil count); its L2 norm is WRMS.
        field=np.linalg.norm(x.reshape(n,3,-1),axis=-1)
        return dict(mean=field.mean(0).tolist(),max=field.max(0).tolist())
    decomposition[stage]=dict(
        forward_linearization_error=error(ideal_linear-actual),
        regularized_inverse_bias=error(inverse_linear-ideal_linear),
        sensor_nonlinearity_transported=error(inverse_nonlinear-inverse_linear),
        total_noiseless_estimator_error=error(inverse_nonlinear-actual),
        propagated_single_packet_noise=error(noise),
        weak_mode_energy_fraction=float(np.sum((states@v[s<.01].T)**2)/np.sum(states**2)))

results={}
for refresh in (False,True):
    for noisy in (False,True):
        label=('refreshed_oracle' if refresh else 'nominal')+('_noisy' if noisy else '_noiseless')
        state=q.copy();trace=[dict(step=0,score=lab.score(state))]
        for it in range(4):
            response=lab.jacobian(state) if refresh else np.broadcast_to(jn,(n,*jn.shape))
            noise=lab.noise(lab.seeds,it) if noisy else np.zeros((n,len(lab.ref)))
            residual=lab.observation(state)-lab.ref+noise
            delta=cap(-(inverse(response)@residual[...,None])[...,0])
            candidates=np.concatenate([lab.hidden+np.clip(state+gain*delta-lab.hidden,-1.5,1.5) for gain in (1,.5,.25)])
            values=lab.observation(candidates).reshape(3,n,-1)-lab.ref+noise[None]
            costs=np.sum(values**2,axis=-1);idx=np.argmin(costs,axis=0)
            improved=costs[idx,np.arange(n)]<np.sum(residual**2,axis=1)
            state=np.where(improved[:,None],candidates.reshape(3,n,35)[idx,np.arange(n)],state)
            trace.append(dict(step=it+1,score=lab.score(state)))
        results[label]=dict(trace=trace,truth_used_to_refresh_response=refresh,
                           noise_design='Paired fixed residual noise per iteration/trial, diagnostic only')
        arrays[label]=state
        print(label,brief(trace[-1]['score']),flush=True)
atomic(HERE/'round2.json',dict(round=2,status='completed',network_updates=0,
    singular_values=s.tolist(),condition_number=float(s[0]/s[-1]),ridge=1e-4,
    weak_mode_count=int((s<.01).sum()),decomposition=decomposition,results=results,
    effective_seconds=active_time()-start))
np.savez_compressed(HERE/'round2_endpoints.npz',**arrays,nominal_native_jacobian=jn,nominal_audit_jacobian=ja)
print('decomposition',json.dumps(decomposition,indent=2),flush=True)
