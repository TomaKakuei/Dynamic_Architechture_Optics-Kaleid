"""Exact archived bounded-WRMS optical term; no new loss components."""
import torch
from input_network import ARMS


class OpticalLoss:
    def __init__(self, optic, arm, out=None):
        assert arm in ARMS
        self.image = None

    def __call__(self, phase, mask, wrms, command, initial, **unused):
        ell = torch.log1p((wrms/.07).square())
        physical = (ell/(1+ell)).mean(-1)
        return physical, dict(bounded_wfe=float(physical.detach().mean()))
