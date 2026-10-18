"""The three one-shot training objectives, including physical validity rules.

Optical propagation stays float64. Ray visibility and path certificates are
discrete masks; their decisions are not differentiated.
"""

import numpy as np
import torch
from torch.nn import functional as F


class NikonObjective:
    def __init__(self, device="cpu"):
        from .physics.nikon import NikonTensorOptics

        self.optics = NikonTensorOptics(device=device)

    def loss(self, command, initial, *, prediction=None, model=None):
        q = initial.double() + command.double()
        with torch.no_grad():
            valid = self.optics.metrics(q)["valid"]
        physical = prediction.sum() * 0
        if valid.any():
            ratio = self.optics.metrics(q[valid])["center"] / 0.07
            physical = (
                torch.sqrt(1 + ratio.square())
                - 1
                + 0.5 * torch.relu(ratio - 1).square()
            ).sum()
        invalid = int((~valid).sum())
        physical = (physical + 100 * invalid) / len(q)
        pose = F.smooth_l1_loss(
            prediction,
            (initial.float() - model.target_mean) / model.target_std,
            beta=0.25,
        )
        loss = physical + 0.02 * pose
        return loss, dict(
            physical=float(physical.detach()),
            pose=float(pose.detach()),
            invalid=invalid,
        )

    @torch.no_grad()
    def metrics(self, command, initial):
        result = self.optics.metrics(initial.double() + command.double())
        return result["center"][:, None].cpu().numpy(), result["valid"].cpu().numpy()


class TrepanObjective:
    def __init__(self, device="cpu"):
        from .physics.trepan import Optics
        from .physics.trepan_geometry import Geometry

        self.center = Optics(grid=65, fields=[0], device=device)
        self.full = Optics(grid=65, device=device)
        self.geometry = Geometry()

    def path_valid(self, q, initial):
        return torch.tensor(
            [
                self.geometry.path(b, a)
                for a, b in zip(
                    initial.detach().cpu().numpy(), q.detach().cpu().numpy()
                )
            ],
            device=q.device,
        )

    def loss(self, command, initial, **kwargs):
        q = initial.double() + command.double()
        wrms, support = self.center.metrics(q)
        wrms, support = wrms[:, 0], support[:, 0]
        ell = torch.log1p((wrms / 0.07).square())
        physical = ell / (1 + ell)
        valid = (support >= 0.98) & torch.isfinite(wrms) & self.path_valid(q, initial)
        physical = torch.where(
            valid, physical, 2 + torch.nan_to_num(physical, nan=1.0, posinf=1.0)
        )
        pose = F.smooth_l1_loss(command, -initial.float(), reduction="none").mean(1)
        mechanical = self.center.mechanical_loss(q)
        loss = (physical + 0.05 * pose + 10 * mechanical).mean()
        return loss, dict(
            physical=float(physical.mean().detach()),
            pose=float(pose.mean().detach()),
            mechanical=float(mechanical.mean().detach()),
            invalid=int((~valid).sum()),
        )

    @torch.no_grad()
    def metrics(self, command, initial):
        q = initial.double() + command.double()
        wrms, support = self.full.metrics(q)
        valid = (
            self.path_valid(q, initial)
            & (support.min(1).values >= 0.98)
            & torch.isfinite(wrms).all(1)
        )
        return wrms.cpu().numpy(), valid.cpu().numpy()


class OlympusObjective:
    def __init__(self, device="cpu"):
        from .physics.olympus import TensorOptics
        from .physics.olympus_geometry import Geometry

        self.center = TensorOptics(device=device, center_only=True)
        self.full = TensorOptics(device=device)
        self.geometry = Geometry()

    def path_valid(self, q, initial):
        result = []
        for end, start in zip(q.detach().cpu().numpy(), initial.detach().cpu().numpy()):
            valid = np.isfinite(end).all() and abs(end).max() <= 1.0
            if valid:
                valid = bool(self.geometry.path(end, start=start)["safe"])
            result.append(valid)
        return torch.tensor(result, dtype=torch.bool, device=q.device)

    def loss(self, command, initial, **kwargs):
        q = initial.double() + command.double()
        mechanical = self.path_valid(q, initial)
        selected = torch.where(mechanical)[0]
        physical = command.sum() * 0.0
        invalid = int((~mechanical).sum())
        if len(selected):
            variance, _, pupil, chief = self.center.high_order(q[selected])
            good = (
                chief.all(1)
                & (pupil.double().mean(-1) >= 0.2).all(1)
                & torch.isfinite(variance).all(1)
            )
            invalid += int((~good).sum())
            if good.any():
                wrms = variance[good, 0].clamp_min(1e-20).sqrt()
                ell = torch.log1p((wrms / 0.07).square())
                physical = (ell / (1 + ell)).sum()
        physical = (physical + 2.0 * invalid) / len(q)
        pose = F.smooth_l1_loss(command, -initial.float())
        domain = torch.relu(q.abs() - 1.0).square().sum(1).mean()
        loss = physical + 0.05 * pose + 10.0 * domain
        return loss, dict(
            physical=float(physical.detach()),
            pose=float(pose.detach()),
            domain=float(domain.detach()),
            invalid=invalid,
        )

    @torch.no_grad()
    def metrics(self, command, initial):
        q = initial.double() + command.double()
        selected = torch.where(self.path_valid(q, initial))[0]
        wrms, valid = np.full((len(q), 9), np.nan), np.zeros(len(q), bool)
        if len(selected):
            variance, _, pupil, chief = self.full.high_order(q[selected])
            w = variance.clamp_min(0).sqrt()
            good = (
                chief.all(1)
                & (pupil.double().mean(-1) >= 0.2).all(1)
                & torch.isfinite(w).all(1)
            )
            ids = selected.cpu().numpy()
            wrms[ids], valid[ids] = w.cpu().numpy(), good.cpu().numpy()
        return wrms, valid


def make_objective(system, device="cpu"):
    return {
        "nikon": NikonObjective,
        "trepan": TrepanObjective,
        "olympus": OlympusObjective,
    }[system](device)


def aggregate(wrms, valid):
    good = valid & np.isfinite(wrms).all(1)
    kept = wrms[good]
    return dict(
        count=len(wrms),
        passes=int(np.sum(good & (wrms[:, 0] < 0.07))),
        invalid=int((~good).sum()),
        recovery_rate=float(np.mean(good & (wrms[:, 0] < 0.07))),
        center_mean=float(kept[:, 0].mean()) if len(kept) else None,
        center_p90=float(np.quantile(kept[:, 0], 0.9)) if len(kept) else None,
        field_mean=float(kept.mean()) if len(kept) else None,
    )


def selection_key(metrics):
    return (
        -metrics["passes"],
        metrics["center_p90"] if metrics["center_p90"] is not None else float("inf"),
        metrics["center_mean"] if metrics["center_mean"] is not None else float("inf"),
    )
