"""FP32 forward-mode optical derivatives, with FP64 residuals elsewhere.

The ray trace is the archived Nikon trace with dtype controlled by its buffers.
Only derivatives use this copy; its cancellation-sensitive phase values are
never used as measured or terminal optical residuals.
"""
import copy
import torch


class FloatTrace:
    def trace(self,effective):
        effective=effective.to(self.bias.dtype)
        transforms=self.transforms(effective)
        p=self.launch_points.expand(len(effective),-1,-1,-1)
        d=self.launch_directions.expand_as(p)
        valid=torch.ones(p.shape[:-1],dtype=torch.bool,device=p.device)
        opl=torch.zeros(p.shape[:-1],dtype=p.dtype,device=p.device)
        for surf in range(1,len(self.path)):
            before,after=self.path[surf-1],self.path[surf]
            rotation,translation=transforms[surf-1]
            bp=torch.einsum('bij,bfrj->bfri',rotation,p-translation[:,None,None])
            bd=torch.einsum('bij,bfrj->bfri',rotation,d)
            perpendicular=-(bp*bd).sum(-1);pp=bp+perpendicular[...,None]*bd
            cv=after['cv'];c=cv*(pp*pp).sum(-1)-2*pp[...,2]
            beta=cv*(bd*pp).sum(-1)-bd[...,2];disc=beta.square()-cv*c
            valid=valid&(disc>0)
            distance=c/(before['zdir']*torch.sqrt(disc.clamp_min(1e-30))-beta)
            hit=pp+distance[...,None]*bd
            normal=-cv*hit+self.axis_z
            normal=normal/torch.linalg.vector_norm(normal,dim=-1,keepdim=True)
            if 1<=surf-1<len(self.path)-2:opl=opl+before['index']*(distance+perpendicular)
            cosine=(bd*normal).sum(-1)
            if after['mode']=='transmit':
                ni,no=before['index'],after['index']
                radicand=no*no-ni*ni*(1-cosine.square());valid=valid&(radicand>=0)
                ncos=torch.copysign(torch.sqrt(radicand.clamp_min(1e-30)),cosine)
                outgoing=(ni*bd+(ncos-ni*cosine)[...,None]*normal)/no
            elif after['mode']=='reflect':outgoing=bd-2*cosine[...,None]*normal
            else:outgoing=bd
            valid=valid&torch.isfinite(hit).all(-1)&torch.isfinite(outgoing).all(-1)
            p,d=hit,outgoing
        return dict(hit=p,direction=d,opl=opl,valid=valid&torch.isfinite(opl))


def make_typed_optics(native,dtype):
    kind=type(native) if isinstance(native,FloatTrace) else type('TypedNikonOptics',(FloatTrace,type(native)),{})
    result=copy.deepcopy(native).to(dtype=dtype);result.__class__=kind
    result.register_buffer('axis_z',torch.tensor([0.,0.,1.],dtype=dtype,device=result.bias.device))
    return result


def make_float_optics(native):return make_typed_optics(native,torch.float32)


def response(optics,physical):
    def one(q):
        traced=optics.trace(q[None]+optics.bias)
        # Subtracting the fixed nominal OPD has exactly zero derivative.
        return (traced['opl'][:,:,1:]@optics.projector.T/optics.wave/7.).flatten()
    # Deployment callers use inference_mode(), which disables forward AD too.
    # Re-enable normal tensors locally for the optical Jacobian only; neither
    # this calculation nor inference builds a neural-parameter backward graph.
    with torch.inference_mode(False),torch.no_grad():
        normal=physical.to(optics.bias.dtype).clone()
        return torch.func.vmap(torch.func.jacfwd(one))(normal).double()
