"""Explicitly authorized4epoch Nikon ideas; separate from stopped long queue."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import argparse,ctypes,os,json,traceback
import numpy as np
import torch
import experiment as exp

class NativeProjectedNetwork(exp.LargerNetwork):
    """Fixed nominal sensor-consistency correction plus trainable full network."""
    def forward(self,image,raw=None):
        prediction=super().forward(image,raw)
        physical=prediction*self.base.target_std+self.base.target_mean
        discrepancy=raw.flatten(1)/7.-physical@self.native_response.T
        corrected=physical+discrepancy@self.native_inverse.T
        return (corrected-self.base.target_mean)/self.base.target_std

class NonlinearProjectedNetwork(exp.LargerNetwork):
    """Nonlinear predicted observation, with bounded physical correction."""
    def forward(self,image,raw=None):
        prediction=super().forward(image,raw)
        physical=(prediction*self.base.target_std+self.base.target_mean).clamp(-1.5,1.5)
        predicted=self.native_optics.metrics(physical.double())['difference'].flatten(1).float()/7.
        delta=(raw.flatten(1)/7.-predicted)@self.native_inverse.T
        delta=delta/torch.clamp(delta.abs().amax(1,keepdim=True)/.25,min=1.)
        return (physical+delta-self.base.target_mean)/self.base.target_std

class LocalModelNetwork(exp.LargerNetwork):
    """One measured input, four model-only nonlinear refinements.

    Exact nonlinear optical innovations use a state-dependent polynomial
    Jacobian fitted on training pairs. No new sensor reading or hidden pose.
    """
    def forward(self,image,raw=None):
        prediction=super().forward(image,raw)
        prior=(prediction*self.base.target_std+self.base.target_mean).clamp(-1.5,1.5).double()
        physical=prior
        measured=raw.flatten(1).double()/7.
        identity=torch.eye(35,dtype=torch.float64,device=physical.device)
        for iteration in range(4):
            observed=self.native_optics.metrics(physical)['difference'].flatten(1)/7.
            jac=(self.poly_linear[None]+2*torch.einsum('fij,bj->bfi',self.poly_quadratic,physical))/7.
            jt=jac.transpose(1,2)
            gram=jt@jac+1e-4*identity
            rhs=(jt@(observed-measured).unsqueeze(-1)).squeeze(-1)+1e-4*(physical-prior)
            delta=-torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
            delta=delta/torch.clamp(delta.abs().amax(1,keepdim=True)/.25,min=1.)
            physical=(physical+delta).clamp(-1.5,1.5)
        return ((physical-self.base.target_mean)/self.base.target_std).float()

class TrialSystem(exp.System):
    def __init__(self,idea):
        super().__init__('nikon');self.idea=idea
        self.warm=exp.HERE/'nikon_large/candidate.pt'
        self.saved=torch.load(self.warm,map_location='cpu',weights_only=False)
        self.trial_config=dict(idea=idea,objective_version=idea+'_short4_v1',
            warm_start_checkpoint=str(self.warm),warm_start_epoch=int(self.saved['epoch']),
            checkpoint_source=str(self.warm.parent),epochs_per_idea=4,
            loss='0.5mean+0.5worst '+idea+' physical loss; pose0.02; invalid100',
            selection_validation_count=6000,full_training_count=24000,
            actual_network_training=True,trainable_parameters=7121929)
        if idea=='pose_anchor':
            self.trial_config.update(loss='Original bounded full-field loss + 1.0 normalized pose SmoothL1; invalid100',
                pose_weight=1.,initial_function_changed=False,
                reason='Test whether original pose weight0.02 leaves an inaccurate inverse despite a well-corrected center')
        if idea=='balanced_pose':
            self.pose_weight=float(json.loads((HERE/'plan.json').read_text())['balanced_pose_weight'])
            self.trial_config.update(loss='Original bounded full-field loss + fixed train-gradient-balanced normalized pose SmoothL1; invalid100',
                pose_weight=self.pose_weight,initial_function_changed=False,
                weight_source=str(HERE/'gradient_balance_design_probe.json'),
                reason='On96original training cases the optical output-gradient norm is14.40858times the unweighted pose gradient; original weight0.02 creates720.43times imbalance')
        if idea=='pose_only':
            self.trial_config.update(loss='Normalized pose SmoothL1 only; unchanged full-field validation and closed-loop scoring',
                pose_weight=1.,physical_weight=0.,initial_function_changed=False,
                reason='Separate inverse-map learning from optical-gradient dominance using exactly4actual epochs and the same full-data split')
        if idea=='closed_loop_training':
            from feedback_training import MeasuredFeedback
            self.feedback=MeasuredFeedback(self)
            self.trial_config.update(loss='Original bounded full-field loss evaluated after actual measured feedback; pose0.02; invalid100',
                inference_protocol='network plus original diff_calibrated measured feedback',
                feedback_rounds=4,reading_budget=20,local_step_cap=.25,early_stop=.04,
                feedback_gains=[1.,.5,.25],feedback_noise_std=.015,initial_function_changed=True,
                attribution='Epoch0 measures adding closed loop; epochs1-4 measure training through that same closed loop',
                reason='One-shot optical gradients prioritize modes corrected by feedback; train against the actual post-feedback residual instead')
        if idea=='sensor_consistency':
            self.native_optics=self.legacy.NikonTensorOptics(nodes=self.legacy.nikon35.NODES,
                fields=self.legacy.nikon35.base.corrected.CONTROL_FIELDS_MM,device='cuda')
            self.trial_config.update(loss='Original bounded full-field loss + pose0.02 + 0.1 nonlinear native-observation pseudo-Huber',
                observation_weight=.1,observation_scale_waves=.07,observation_fields_mm=[0.,-.05,.05],
                initial_function_changed=False,inference_projection=False,
                reason='Penalize measured nonlinear sensor disagreement during training without a fixed inverse or inference correction')
        if idea=='nonlinear_local_model':
            self.native_optics=self.legacy.NikonTensorOptics(nodes=self.legacy.nikon35.NODES,
                fields=self.legacy.nikon35.base.corrected.CONTROL_FIELDS_MM,device='cuda')
            with np.load(HERE/'native_polynomial.npz') as z:
                self.poly_linear=torch.tensor(z['linear'],dtype=torch.float64,device='cuda')
                self.poly_quadratic=torch.tensor(z['quadratic'],dtype=torch.float64,device='cuda')
            self.trial_config.update(loss='Original bounded full-field loss after4differentiable model-only refinements; pose0.02; invalid100',
                inference_protocol='one acquisition: network plus4model-only nonlinear refinements',
                measurement_acquisitions=1,model_refinements=4,refinement_ridge=1e-4,step_cap=.25,
                jacobian_surrogate=str(HERE/'native_polynomial.npz'),surrogate_fit_count=24000,
                exact_nonlinear_innovation=True,initial_function_changed=True,
                attribution='Epoch0 measures refinement; epochs1-4 measure subsequent full-network learning',
                reason='Nominal Jacobian failed far from bias; use state-dependent approximate Jacobian with exact nonlinear innovations from the same single measured packet')
        if idea=='exact_local_response':
            neural_keys=set(self.saved['model'])
            self.warm=HERE/'nonlinear_local_model/candidate.pt'
            previous=torch.load(self.warm,map_location='cpu',weights_only=False)
            self.saved=dict(previous,model={k:v for k,v in previous['model'].items() if k in neural_keys})
            self.native_optics=self.legacy.NikonTensorOptics(nodes=self.legacy.nikon35.NODES,
                fields=self.legacy.nikon35.base.corrected.CONTROL_FIELDS_MM,device='cuda')
            from fp32_response import make_typed_optics
            self.native_optics=make_typed_optics(self.native_optics,torch.float64)
            self.trial_config.update(loss='Original bounded full-field loss after4exact-response model-only refinements; pose0.02; invalid100',
                inference_protocol='one acquisition: network plus4exact-response nonlinear model refinements',
                measurement_acquisitions=1,model_refinements=4,refinement_ridge=1e-4,step_cap=.25,
                response_backend='FP64 forward-mode automatic differentiation; NN parameters FP32',
                warm_start_checkpoint=str(self.warm),warm_start_epoch=int(self.saved['epoch']),
                checkpoint_source=str(self.warm.parent),
                model_gains=[1.,.5,.25],initial_function_changed=True,
                backward='Implicit Gauss-Newton at terminal pose; approximate derivative, not exact four-step trajectory derivative',
                attribution='Epoch0 measures exact-response optimizer; epochs1-4 measure subsequent network learning',
                reason='Continue the successful19.17percent candidate and remove its quadratic response-approximation error; compare against its own epoch0')
        if idea in ('native_projection','nonlinear_projection'):
            native=self.legacy.NikonTensorOptics(nodes=self.legacy.nikon35.NODES,
                fields=self.legacy.nikon35.base.corrected.CONTROL_FIELDS_MM,device='cuda')
            self.native_optics=native
            dq=torch.eye(35,dtype=torch.float64,device='cuda')*1e-4
            with torch.no_grad():
                a=native.metrics(dq)['difference'].flatten(1)/7.
                b=native.metrics(-dq)['difference'].flatten(1)/7.
                j=((a-b)/(2e-4)).T.cpu().numpy()
            u,s,v=np.linalg.svd(j,full_matrices=False)
            self.native_response=torch.tensor(j,dtype=torch.float32,device='cuda')
            self.native_inverse=torch.tensor((v.T*(s/(s*s+1e-4)))@u.T,dtype=torch.float32,device='cuda')
            self.trial_config.update(loss='bounded control + fixed differentiable native-observation projection',
                projection_fields_mm=[0.,-.05,.05],projection_ridge=1e-4,
                projection_dtype='float32',initial_function_changed=True,
                attribution='Epoch0 measures engineering projection; epochs1-4 measure subsequent learning')
            if idea=='nonlinear_projection':
                self.trial_config.update(loss='bounded control + differentiable nonlinear sensor innovation and0.25 correction cap',
                    correction_max_coordinate=.25,
                    reason='Prior fixed linear prediction produced19.79waves measurement-model error and835.75coordinate correction on16training cases')

    def model(self,variant):
        model=super().model(variant)
        model.load_state_dict(self.saved['model']);model.requires_grad_(True)
        if self.idea in ('native_projection','nonlinear_projection'):
            model.__class__=NativeProjectedNetwork if self.idea=='native_projection' else NonlinearProjectedNetwork
            model.register_buffer('native_response',self.native_response)
            model.register_buffer('native_inverse',self.native_inverse)
            if self.idea=='nonlinear_projection':model.native_optics=self.native_optics
        if self.idea=='nonlinear_local_model':
            model.__class__=LocalModelNetwork
            model.native_optics=self.native_optics
            model.register_buffer('poly_linear',self.poly_linear)
            model.register_buffer('poly_quadratic',self.poly_quadratic)
        if self.idea=='exact_local_response':
            from exact_refinement import ExactResponseNetwork
            model.__class__=ExactResponseNetwork
            model.native_optics=self.native_optics
        return model

    def forward(self,model,data,indices):
        endpoint,initial,command,pose=super().forward(model,data,indices)
        if self.idea=='closed_loop_training':
            raw=torch.as_tensor(data['raw'][indices],device=command.device)
            command=self.feedback.run(initial,command,data['ids'][indices],raw)
            endpoint=initial+command
        return endpoint,initial,command,pose

    def loss(self,model,data,indices):
        if self.idea=='pseudohuber':return super().loss(model,data,indices)
        endpoint,initial,command,pose=self.forward(model,data,indices)
        if self.idea=='pose_only':
            return pose,dict(physical=0.,pose=float(pose.detach()),mechanical=0.,invalid=0)
        with torch.no_grad():_,valid,_=self.metrics(endpoint,initial)
        if bool(valid.any()):
            fields,_,_=self.metrics(endpoint[valid],initial[valid])
            ell=torch.log1p((fields/self.threshold).square());term=ell/(1+ell)
            physical=((.5*term.mean(1)+.5*term.amax(1)).sum()+100*(~valid).sum())/len(indices)
        else:physical=command.sum()*0+100.
        observation=command.sum()*0
        observation_invalid=0
        if self.idea=='sensor_consistency':
            raw=torch.as_tensor(data['raw'][indices],device=command.device)
            with torch.no_grad():
                observed=self.native_optics.metrics(-command.double())
                usable=observed['valid'] & torch.isfinite(observed['difference']).flatten(1).all(1)
            observation_invalid=int((~usable).sum())
            if bool(usable.any()):
                predicted=self.native_optics.metrics(-command[usable].double())['difference']
                residual_rms=((predicted-raw[usable]).square().mean(-1)+1e-20).sqrt()
                term=(torch.sqrt(1+(residual_rms/.07).square())-1).mean(1)
                observation=(term.sum()+100*(~usable).sum())/len(indices)
            else:observation=command.sum()*0+100.
        pose_weight=getattr(self,'pose_weight',1. if self.idea=='pose_anchor' else .02)
        loss=physical+pose_weight*pose+.1*observation
        return loss,dict(physical=float(physical.detach()),pose=float(pose.detach()),
            observation=float(observation.detach()),observation_invalid=observation_invalid,
            mechanical=0.,invalid=int((~valid).sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--idea',choices=['bounded_control','pseudohuber','native_projection','nonlinear_projection','pose_anchor','sensor_consistency','balanced_pose','pose_only','closed_loop_training','nonlinear_local_model','exact_local_response'],required=True);a=p.parse_args()
    authorization=json.loads((HERE/'AUTHORIZATION.json').read_text())
    assert authorization['epochs_per_idea']==4 and authorization['full_network_training']
    args=argparse.Namespace(stage='train',system='nikon',variant='large',epochs=4,batch=128,microbatch=128,seed=202610241)
    out=HERE/a.idea;out.mkdir(exist_ok=True)
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        system=TrialSystem(a.idea)
        exp.train(system,args,out=out)
        final=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        differences=[(final['model'][k].double()-v.double()).square().sum().item()
                     for k,v in system.saved['model'].items() if v.is_floating_point()]
        exp.atomic(out/'training_evidence.json',dict(completed_epochs=4,training_cases_per_epoch=24000,
            optimizer_updates=4*int(np.ceil(24000/128)),state_delta_l2=float(np.sqrt(sum(differences))),
            all_parameters_trainable=True,network_dtype='float32',source_checkpoint=str(system.warm)))
        if a.idea=='closed_loop_training':
            direct=TrialSystem('bounded_control');model=direct.model('large')
            model.load_state_dict(final['model']);data=direct.load_data()
            result=exp.evaluate(direct,model,data,np.flatnonzero(~data['train']),128,out/'one_shot_validation_004.npz')
            exp.atomic(out/'one_shot_summary.json',dict(epoch=4,validation=result,
                scope='Same trained network before feedback; distinct from full closed-loop validation'))
    except Exception:
        exp.atomic(out/'failure.json',dict(error=traceback.format_exc(),updated=exp.stamp()));raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
