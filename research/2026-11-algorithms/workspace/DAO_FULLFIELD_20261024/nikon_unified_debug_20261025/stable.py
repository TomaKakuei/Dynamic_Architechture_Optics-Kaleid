"""Conditioned full-pose head, one shared network, all parameters trainable."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
REPAIR=HERE.parent/'nikon_unified_repair_20261024'
sys.path.insert(0,str(REPAIR))
from repair import RepairSystem,FullPoseTrunk,ExactResponseNetwork,exp,torch,np,SEED,PARAMETERS,SHORT

class NormalizedTrunk(FullPoseTrunk):
    def __init__(self,width,source):
        super().__init__(width,source)
        self.register_buffer('feature_center',torch.zeros(512))
        self.register_buffer('feature_scale',torch.ones(512))

    def forward(self,image,raw=None):
        feature=self.features(image,raw)
        prediction=self.head[-1]((feature-self.feature_center)/self.feature_scale)
        if torch.is_grad_enabled():self.prior_prediction=prediction
        return prediction

class NormalizedExact(ExactResponseNetwork,NormalizedTrunk):
    pass

class StableSystem(RepairSystem):
    def __init__(self):
        super().__init__()
        self.decision=json.loads((HERE/'decision.json').read_text())
        assert self.decision['passed']
        self.initialization=HERE/self.decision['initialization_checkpoint']
        self.trial_config.update(idea='unified_conditioned_readout',
            objective_version='unified_explicit_feature_normalization_update_scale_v1',
            warm_start_checkpoint=str(self.initialization),warm_start_epoch=0,
            checkpoint_source=str(self.initialization),
            initialization=self.decision['initialization_description'],
            initialization_ridge=self.decision['ridge'],
            initialization_variance_floor=self.decision.get('feature_std_floor'),
            optimizer=f"All 101 FP32 tensors AdamW backbone lr={self.decision['learning_rate']}, readout lr={self.decision['readout_learning_rate']}; same four-epoch cosine",
            learning_rate=self.decision['learning_rate'],
            reason='Keep training-feature centering/scaling explicit instead of folding them into enormous FP32 readout weights; constrain sensitive backbone update scale',
            diagnostic_discarded_optimizer_updates=self.decision['diagnostic_discarded_optimizer_updates'],
            attribution='Explicit readout parameterization and learning-rate fix; original terminal and direct-prior losses unchanged',
            validation_used_for_repair_selection=False)

    def fresh_model(self):
        model=NormalizedExact(1648,self.source_weights).cuda()
        model.native_optics=self.native_optics
        assert sum(p.numel() for p in model.parameters())==PARAMETERS
        assert {p.dtype for p in model.parameters()}=={torch.float32}
        return model

    def model(self,variant):
        assert variant=='large'
        model=self.fresh_model()
        model.load_state_dict(torch.load(self.initialization,map_location='cpu',weights_only=False)['model'])
        return model

    def loss(self,model,data,indices):
        loss,parts=super().loss(model,data,indices)
        model._update_probe=(torch.as_tensor(data['image'][indices],device='cuda'),
            torch.as_tensor(data['raw'][indices],device='cuda'),model.prior_prediction.detach())
        return loss,parts

class ObservedAdamW(torch.optim.AdamW):
    """Record actual neural-output changes; optimizer math is unchanged."""
    def __init__(self,groups,model):
        super().__init__(groups,weight_decay=1e-4)
        self.observed_model=model

    def step(self,closure=None):
        result=super().step(closure)
        step=int(next(iter(self.state.values()))['step'])
        if (step<=3 or step%10==0) and hasattr(self.observed_model,'_update_probe'):
            image,raw,before=self.observed_model._update_probe
            with torch.no_grad():
                after=NormalizedTrunk.forward(self.observed_model,image,raw)
                assert torch.isfinite(after).all()
                entry=dict(step=step,updated=exp.stamp(),
                    neural_prediction_delta_rms=float((after-before).square().mean().sqrt()),
                    neural_prediction_delta_max=float((after-before).abs().max()),
                    readout_weight_norm=float(self.observed_model.head[-1].weight.norm()),
                    learning_rates=[g['lr'] for g in self.param_groups])
            with (HERE/'update_diagnostics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(entry)+'\n')
        return result

def optimizer(model):
    decision=json.loads((HERE/'decision.json').read_text())
    lr=decision['learning_rate'];head_lr=decision['readout_learning_rate']
    readout=list(model.head[-1].parameters());ids={id(p) for p in readout}
    return ObservedAdamW([dict(params=[p for p in model.parameters() if id(p) not in ids],lr=lr,initial_lr=lr),
        dict(params=readout,lr=head_lr,initial_lr=head_lr)],model)
