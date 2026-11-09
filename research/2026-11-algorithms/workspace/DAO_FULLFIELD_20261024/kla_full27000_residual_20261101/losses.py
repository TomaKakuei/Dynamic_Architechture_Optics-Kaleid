"""Unchanged archived bounded-WRMS objective for all three experiments."""
import torch
from models import ARMS

class OpticalLoss:
    def __init__(self, optic, arm, out=None):
        self.image = None
        assert arm in ARMS

    def __call__(self, phase, mask, wrms, command, initial, **unused):
        ell = torch.log1p((wrms/.07).square())
        loss = (ell/(1+ell)).mean(-1)
        return loss,dict(bounded_wfe=float(loss.detach().mean()))
