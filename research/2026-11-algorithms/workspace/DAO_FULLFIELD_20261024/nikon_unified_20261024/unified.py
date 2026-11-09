"""Single shared Nikon backbone, two sizes, identical one-acquisition optics."""
from pathlib import Path
import sys,copy,json
HERE=Path(__file__).resolve().parent
SHORT=HERE.parent/'nikon_short_training'
sys.path.insert(0,str(SHORT))
from short_train import TrialSystem,exp,torch,np
from exact_refinement import ExactResponseNetwork
nn=torch.nn
SOURCE=SHORT/'exact_local_response/candidate.pt'
REFERENCE_PARAMETERS=7121929

class UnifiedTrunk(exp.LargerNetwork):
    """No grouped encoders or group-specific heads; one shared35D predictor."""
    def __init__(self,feedforward_width,source):
        nn.Module.__init__(self)
        self.system='nikon'
        self.base=nn.Module()  # Normalization buffers only; no base network.
        for name in ('raw_mean','raw_std','target_mean','target_std'):
            self.base.register_buffer(name,source['base.'+name].clone())
        layers=[];incoming=1
        for channels in (32,64,128,256,512):
            layers.extend([nn.Conv2d(incoming,channels,3,stride=2,padding=1),
                nn.GroupNorm(8,channels),nn.GELU(),
                nn.Conv2d(channels,channels,3,padding=1,groups=channels),
                nn.Conv2d(channels,channels,1),nn.GELU()])
            incoming=channels
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1))
        self.projection=nn.Linear(512,256)
        self.field_identity=nn.Parameter(torch.zeros(1,3,256))
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(256,8,feedforward_width,
            dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(4)])
        self.raw_branch=nn.Sequential(nn.Linear(147,512),nn.GELU(),nn.Linear(512,256))
        self.head=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,512),nn.GELU(),nn.Linear(512,35))
        self.transfer_global_branch(source)

    @torch.no_grad()
    def transfer_global_branch(self,source):
        """Copy all compatible global tensors; added FFN units start neutral."""
        target=self.state_dict()
        for key,value in target.items():
            old=source[key]
            if old.shape==value.shape:value.copy_(old)
            elif '.linear1.weight' in key or '.linear1.bias' in key:
                assert value.shape[0]>=old.shape[0]
                value[:old.shape[0]].copy_(old)
            elif '.linear2.weight' in key:
                assert value.shape[1]>=old.shape[1]
                value.zero_();value[:,:old.shape[1]].copy_(old)
            else:raise RuntimeError(f'Unexpected transfer shape:{key}:{old.shape}->{value.shape}')

    def forward(self,image,raw=None):
        normalized=torch.asinh(image)
        batch,fields,height,width=normalized.shape
        tokens=self.projection(self.encoder(normalized.reshape(batch*fields,1,height,width)).flatten(1))
        tokens=tokens.reshape(batch,fields,256)+self.field_identity
        native=((raw-self.base.raw_mean)/self.base.raw_std).flatten(1)
        tokens=tokens+self.raw_branch(native)[:,None]
        for layer in self.layers:tokens=layer(tokens)
        return self.head(tokens.flatten(1))

class UnifiedExactNetwork(ExactResponseNetwork,UnifiedTrunk):
    # ExactResponseNetwork.forward calls super().forward, which the MRO sends
    # to UnifiedTrunk.forward. Its four-step optical solver stays unchanged.
    pass

def layout():
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)['model']
    torch.manual_seed(202610241)
    probe=UnifiedTrunk(1024,source)
    base=sum(p.numel() for p in probe.parameters())
    increment=4*(2*256+1)
    configs={}
    for size,multiplier in [('1x',1),('2x',2)]:
        width=1024+8*round((REFERENCE_PARAMETERS*multiplier-base)/(increment*8))
        count=base+(width-1024)*increment
        assert width>=1024 and abs(count/REFERENCE_PARAMETERS/multiplier-1)<.002
        configs[size]=dict(feedforward_width=width,parameter_count=count,
            ratio_to_original=count/REFERENCE_PARAMETERS)
    return configs

class UnifiedSystem(TrialSystem):
    def __init__(self,size):
        super().__init__('exact_local_response')
        self.size=size
        self.spec=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))['models'][size]
        self.global_source=torch.load(SOURCE,map_location='cpu',weights_only=False)['model']
        self.trial_config.update(idea='unified_'+size,objective_version='unified_single_acquisition_'+size+'_v1',
            inference_protocol='One observation packet, one unified35D network,4unchanged exact-response model steps, no measured feedback',
            architecture='Shared field CNN,4shared256D attention blocks,raw147branch,single35Dhead; no grouped subnets',
            parameter_count=self.spec['parameter_count'],trainable_parameters=self.spec['parameter_count'],
            feedforward_width=self.spec['feedforward_width'],
            initialization='Same global branch from exact_local_response epoch0; grouped encoders and coordination removed; new FFN output columns zero',
            source_checkpoint=str(SOURCE),checkpoint_source=str(SOURCE.parent),warm_start_checkpoint=str(SOURCE),warm_start_epoch=0,
            initialization_limit='Preserves transferred global-branch prediction, not the complete old modular predictor',
            optimizer='AdamW all neural weights lr1e-4,decay1e-4,same4epoch cosine; empty normalization-only base group',
            training_conditions='Same24000training samples and order,6000validation,4epochs,batch128,FP32 parameters',
            measured_feedback=False,model_refinements=4,
            attribution='Compare own epoch0 and epoch4; sizes share the same initial neural function up to floating-point arithmetic')

    def model(self,variant):
        assert variant=='large'
        model=UnifiedExactNetwork(self.spec['feedforward_width'],self.global_source).cuda()
        model.native_optics=self.native_optics
        assert not list(model.base.parameters())
        assert sum(p.numel() for p in model.parameters())==self.spec['parameter_count']
        assert {p.dtype for p in model.parameters()}=={torch.float32}
        assert model.__class__.__mro__[1:4]==(ExactResponseNetwork,UnifiedTrunk,exp.LargerNetwork)
        return model

if __name__=='__main__':print(json.dumps(layout(),indent=2))
