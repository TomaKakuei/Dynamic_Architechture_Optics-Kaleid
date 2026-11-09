"""FP32 optical derivatives with detached accurate reference wavefront values."""
import torch


def residual(optical, q, fixed_mask=None):
    phase, traced_mask = optical.phase(q)
    mask = traced_mask if fixed_mask is None else fixed_mask
    weight = mask.to(phase.dtype)
    design = optical.design
    gram = torch.einsum('nr,bfn,ns->bfrs', design, weight, design)
    rhs = torch.einsum('nr,bfn->bfr', design, weight*phase)
    eye = torch.eye(4, dtype=phase.dtype, device=phase.device)
    coefficient = torch.linalg.solve(gram+eye*1e-12, rhs[..., None])[..., 0]
    value = (phase-torch.einsum('nr,bfr->bfn', design, coefficient))*weight
    return value, mask


def wrms(value, mask):
    return torch.sqrt(value.square().sum(-1)/mask.sum(-1).clamp_min(1)+1e-20)


def corrected_fp32_metrics(optical32, q, reference_residual, reference_mask):
    approximate, _ = residual(optical32, q.float(), reference_mask)
    # Accurate reference values do not retain an FP64 backward graph. Optical
    # Jacobians and the trainable path below are FP32. Validation stays native.
    value = reference_residual.float()+(approximate-approximate.detach())
    return wrms(value, reference_mask)
