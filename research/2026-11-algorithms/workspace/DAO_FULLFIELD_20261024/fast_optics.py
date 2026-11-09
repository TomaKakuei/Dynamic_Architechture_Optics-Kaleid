"""Same float64 Trepan optics, explicit 3-vector rotations instead of skinny BMM."""
import numpy as np
import torch
from optics import Optics, GROUPS


def rotate(matrix, vector, transpose=False):
    # Preserve the three-term dot product; avoid generic GEMM and its large
    # reduction GEMMs in backward for the many tiny3x3 rotation matrices.
    if transpose:
        matrix = matrix.transpose(1, 2)
    return torch.stack([
        matrix[:, i, 0, None, None]*vector[..., 0]
        + matrix[:, i, 1, None, None]*vector[..., 1]
        + matrix[:, i, 2, None, None]*vector[..., 2]
        for i in range(3)], dim=-1)


class FastOptics(Optics):
    def trace(self, q):
        center, rots = self.poses(q)
        p = self.p0.expand(len(q), -1, -1, -1)
        d = self.d0.expand_as(p)
        op = self.o0.expand(len(q), -1, -1)
        valid = self.v0.expand(len(q), -1, -1)
        for s in self.surfaces:
            gi = next(i for i, (a, b) in enumerate(GROUPS) if a <= s['sid'] <= b)
            r = rots[:, gi]
            origin = center[:, gi]+r[:, :, 2]*(s['z']-self.pivots[gi, 2])
            bp = rotate(r, p-origin[:, None, None], transpose=True)
            bd = rotate(r, d, transpose=True)
            cv = 0 if np.isinf(s['r']) else 1/s['r']
            perp = -(bp*bd).sum(-1)
            pp = bp+perp[..., None]*bd
            c = cv*(pp*pp).sum(-1)-2*pp[..., 2]
            beta = cv*(bd*pp).sum(-1)-bd[..., 2]
            disc = beta*beta-cv*c
            valid = valid & (disc>0)
            t = perp+c/(torch.sqrt(disc.clamp_min(1e-30))-beta)
            hit = bp+t[..., None]*bd
            n = -cv*hit+hit.new_tensor([0., 0., 1.])
            n = n/torch.linalg.vector_norm(n, dim=-1, keepdim=True)
            valid = valid & (hit[..., :2].square().sum(-1)<=(s['sd']+1e-9)**2)
            op = op+s['ni']*t
            cos = (bd*n).sum(-1)
            rad = s['no']**2-s['ni']**2*(1-cos*cos)
            valid = valid & (rad>=0)
            dd = (s['ni']*bd+(torch.copysign(torch.sqrt(rad.clamp_min(1e-30)), cos)-s['ni']*cos)[..., None]*n)/s['no']
            p = rotate(r, hit)+origin[:, None, None]
            d = rotate(r, dd)
        return p, d, op, valid
