"""Full-pose readout repair of the 7,121,987-parameter shared Nikon network."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'nikon_unified_20261024'
sys.path.insert(0,str(PREVIOUS))
from unified import UnifiedTrunk,ExactResponseNetwork,TrialSystem,exp,torch,np,SOURCE,SHORT
nn=torch.nn
SEED=202610241
PARAMETERS=7121987

class FullPoseTrunk(UnifiedTrunk):
    def features(self,image,raw):
        normalized=torch.asinh(image)
        batch,fields,height,width=normalized.shape
        tokens=self.projection(self.encoder(normalized.reshape(batch*fields,1,height,width)).flatten(1))
        tokens=tokens.reshape(batch,fields,256)+self.field_identity
        native=((raw-self.base.raw_mean)/self.base.raw_std).flatten(1)
        tokens=tokens+self.raw_branch(native)[:,None]
        for layer in self.layers:tokens=layer(tokens)
        return self.head[:-1](tokens.flatten(1))

    def forward(self,image,raw=None):
        prediction=self.head[-1](self.features(image,raw))
        # Training-only access to the neural prior, before clipping/optical solve.
        if torch.is_grad_enabled():self.prior_prediction=prediction
        return prediction

class RepairedNetwork(ExactResponseNetwork,FullPoseTrunk):
    pass

class RepairSystem(TrialSystem):
    def __init__(self):
        super().__init__('exact_local_response')
        self.source_weights=torch.load(SOURCE,map_location='cpu',weights_only=False)['model']
        self.trial_config.update(idea='unified_complete_pose_readout',
            objective_version='unified_complete_pose_readout_direct_supervision_v1',
            architecture='One shared field CNN, four shared attention blocks, one 35D full-pose head',
            trainable_parameters=PARAMETERS,parameter_count=PARAMETERS,
            checkpoint_source=str(SOURCE),warm_start_checkpoint=str(HERE/'initialized.pt'),
            warm_start_epoch=0,source_checkpoint=str(SOURCE),
            initialization='Fixed ridge readout fit on all 24000 training labels; replaces residual-only output head',
            initialization_is_training=True,initialization_optimizer_updates=0,
            initialization_fit_samples=24000,initialization_ridge=0.001,
            loss='Original bounded full-field terminal loss + 0.02 terminal pose + 1.0 direct neural-prior pose SmoothL1(beta=0.25)',
            direct_prior_pose_weight=1.,all_parameters_trainable=True,
            measurement_acquisitions=1,measured_feedback=False,model_refinements=4,
            inference_inputs='image and raw from the same archived observation packet; no labels or teacher',
            inference_protocol='One unified 35D prediction plus unchanged four model-only exact-response steps',
            validation_protocol='6000 original cases, center ABS and edge DIFF, all three below 0.07 waves; invalids fail',
            optimizer='All 101 FP32 parameter tensors, AdamW 1e-4, same four-epoch cosine schedule',
            attribution='Combined readout initialization and direct-prior supervision repair; not an isolated ablation')

    def fresh_model(self):
        model=RepairedNetwork(1648,self.source_weights).cuda()
        model.native_optics=self.native_optics
        assert sum(p.numel() for p in model.parameters())==PARAMETERS
        assert not list(model.base.parameters())
        assert {p.dtype for p in model.parameters()}=={torch.float32}
        return model

    def model(self,variant):
        assert variant=='large'
        model=self.fresh_model()
        model.load_state_dict(torch.load(HERE/'initialized.pt',map_location='cpu',weights_only=False)['model'])
        return model

    def loss(self,model,data,indices):
        loss,parts=super().loss(model,data,indices)
        target=torch.as_tensor(data['q'][indices],dtype=torch.float32,device='cuda')
        target=(target-model.base.target_mean)/model.base.target_std
        direct=nn.functional.smooth_l1_loss(model.prior_prediction,target,beta=.25)
        parts['direct_prior_pose']=float(direct.detach())
        return loss+direct,parts
