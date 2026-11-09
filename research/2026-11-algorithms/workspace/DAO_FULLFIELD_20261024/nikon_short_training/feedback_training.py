"""Differentiable batched implementation of the existing Nikon measured loop.

The controller receives noisy center ABS/native DIFF packets only. Hidden
poses are used solely inside the simulated sensor and terminal training loss.
The 20-reading ceiling, four rounds, gains, step cap and stop are unchanged.
"""
import numpy as np
import torch


class MeasuredFeedback:
    def __init__(self,system):
        from nikon_closed_loop import NikonFeedback
        reference=NikonFeedback()
        self.native=reference.native
        self.center=system.legacy.NikonTensorOptics(nodes=system.legacy.nikon35.AUDIT_NODES,
            fields=(0.,),device='cuda')
        self.center_count=len(self.center.nodes)
        self.audit_count=len(system.optics.nodes)
        for key in ('nominal_inverse','observed_reference','reported_reference','reported_response'):
            setattr(self,key,torch.tensor(getattr(reference,key),dtype=torch.float64,device='cuda'))
        self.last_counts=None

    def center_signal(self,q):
        # Same reference-sphere PTTD calculation as NikonTensorOptics.metrics,
        # with one field so no unused edge tracing or empty edge reduction.
        o=self.center;t=o.trace(q.double()+o.bias)
        hit=t['hit'][:,:,1:];direction=t['direction'][:,:,1:]
        direction=direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
        delta=hit-t['hit'][:,:,:1]
        projected=(delta*direction).sum(-1)
        radicand=projected.square()-delta.square().sum(-1)+100.
        sphere=t['opl'][:,:,1:]-projected-torch.sqrt(radicand.clamp_min(1e-30))
        absolute=(sphere@o.projector.T/o.wave)[:,0]
        valid=t['valid'].all(-1).all(-1)&(radicand>0).all(-1).all(-1)
        return absolute,valid

    def noise_tape(self,seeds):
        # Match Plant.measure's native-then-center draw order for each case.
        size=147+self.center_count
        tape=np.stack([np.random.RandomState(int(seed)).normal(0.,.015,(20,size)) for seed in seeds])
        return torch.as_tensor(tape,device='cuda')

    def estimate(self,observation):
        return (observation-self.observed_reference)@self.nominal_inverse.T

    def merit(self,observation):
        center=torch.linalg.vector_norm(observation[:,:self.center_count],dim=1)
        report=(self.reported_reference+self.estimate(observation)@self.reported_response.T)
        report=report.reshape(-1,3,self.audit_count)
        transported=torch.sqrt(report.square().mean(-1)).amax(1)
        return torch.maximum(center,transported)

    def run(self,initial,predicted_command,seeds,initial_raw):
        batch=len(initial);device=initial.device
        counts=torch.zeros(batch,dtype=torch.long,device=device)
        noise=self.noise_tape(seeds)
        all_rows=torch.arange(batch,device=device)

        def measure(command,rows,provided_raw=None):
            q=initial[rows]+command
            native=self.native.metrics(q)
            center,center_valid=self.center_signal(q)
            perturbation=noise[rows,counts[rows]]
            raw=native['difference'].flatten(1)+perturbation[:,:147]
            if provided_raw is not None:raw=provided_raw.flatten(1).double()
            center=center+perturbation[:,147:]
            observation=torch.cat((center/np.sqrt(self.center_count),raw/7.),dim=1)
            valid=native['valid']&center_valid&torch.isfinite(observation).all(1)
            merit=torch.where(valid,self.merit(observation),torch.full_like(center[:,0],float('inf')))
            counts[rows]+=1
            return observation,merit

        command=torch.zeros_like(initial)
        observation,best=measure(command,all_rows,initial_raw)
        proposed=predicted_command.double()
        candidate_observation,candidate_merit=measure(proposed,all_rows)
        accept=candidate_merit<best
        if bool(accept.any()):
            command=torch.where(accept[:,None],proposed,command)
            observation=torch.where(accept[:,None],candidate_observation,observation)
            best=torch.where(accept,candidate_merit,best)
        running=torch.isfinite(best)
        for iteration in range(4):
            active=running&(best>.04)&(counts<20)
            if not bool(active.any()):break
            direction=-self.estimate(observation)
            direction=direction/torch.clamp(direction.abs().amax(1,keepdim=True)/.25,min=1.)
            origin=command
            accepted=torch.zeros_like(active)
            for gain in (1.,.5,.25):
                rows=torch.nonzero(active&~accepted&(counts<20),as_tuple=True)[0]
                if len(rows)==0:break
                trial=(origin[rows]+gain*direction[rows]).clamp(-1.5,1.5)
                trial_observation,trial_merit=measure(trial,rows)
                local_accept=trial_merit<best[rows]
                if bool(local_accept.any()):
                    selected=rows[local_accept]
                    command=command.index_copy(0,selected,trial[local_accept])
                    observation=observation.index_copy(0,selected,trial_observation[local_accept])
                    best=best.index_copy(0,selected,trial_merit[local_accept])
                    accepted[selected]=True
            running=active&accepted
        self.last_counts=counts.detach().cpu().numpy()
        return command
