"""Float64 Nikon ray propagation and absolute/bias-referenced PTTD metrics."""

import numpy as np
import torch
from torch import nn
from .assets import load_spec, load_buffers


class NikonTensorOptics(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        spec = load_spec("nikon_optics")
        self.nodes = np.asarray(spec["nodes"])
        self.fields = tuple(spec["fields"])
        self.wave = spec["wave"]
        self.groups = tuple(map(tuple, spec["groups"]))
        self.path, self.gaps, self.spans = spec["path"], spec["gaps"], spec["spans"]
        self.ends = {end: i for i, (_, end) in enumerate(self.groups)}
        self.starts = {start: i for i, (start, _) in enumerate(self.groups)}
        load_buffers(self, "nikon_optics", device)

    def rotations(self, effective):
        state = effective.reshape(-1, 7, 5) * self.scales
        # RayOptics optical Euler convention: intrinsic Rx(-alpha) Ry(-beta).
        x, y = -torch.deg2rad(state[..., 3]), -torch.deg2rad(state[..., 4])
        cx, sx, cy, sy = torch.cos(x), torch.sin(x), torch.cos(y), torch.sin(y)
        z, o = torch.zeros_like(x), torch.ones_like(x)
        rot = torch.stack(
            (cy, z, sy, sx * sy, cx, -sx * cy, -cx * sy, sx, cx * cy), -1
        ).reshape(-1, 7, 3, 3)
        return state, rot

    def transforms(self, effective):
        state, rotations = self.rotations(effective)
        identity = torch.eye(3, dtype=effective.dtype, device=effective.device).expand(
            len(effective), -1, -1
        )
        transforms = []
        for before, gap in enumerate(self.gaps):
            translation = torch.zeros(
                (len(effective), 3), dtype=effective.dtype, device=effective.device
            )
            translation[:, 2] = gap
            after_rotation = before_rotation = None
            if before in self.ends:
                group = self.ends[before]
                r = rotations[:, group]
                span = torch.zeros_like(translation)
                span[:, 2] = self.spans[group]
                leave = state[:, group, :3] + torch.einsum(
                    "bij,bj->bi", r - identity, span
                )
                translation = translation - leave
                after_rotation = r.transpose(-1, -2)
            if before + 1 in self.starts:
                group = self.starts[before + 1]
                translation = translation + state[:, group, :3]
                before_rotation = rotations[:, group]
            rotation = identity
            if after_rotation is not None:
                translation = torch.einsum("bij,bj->bi", after_rotation, translation)
                rotation = after_rotation
                if before_rotation is not None:
                    rotation = rotation @ before_rotation
            elif before_rotation is not None:
                rotation = before_rotation
            transforms.append((rotation.transpose(-1, -2), translation))
        return transforms

    def trace(self, effective):
        if effective.ndim != 2 or effective.shape[1] != 35:
            raise ValueError("Expected batch x35 effective normalized states")
        effective = effective.to(torch.float64)
        transforms = self.transforms(effective)
        p = self.launch_points.expand(len(effective), -1, -1, -1)
        d = self.launch_directions.expand_as(p)
        valid = torch.ones(p.shape[:-1], dtype=torch.bool, device=p.device)
        opl = torch.zeros(p.shape[:-1], dtype=p.dtype, device=p.device)
        for surf in range(1, len(self.path)):
            before, after = self.path[surf - 1], self.path[surf]
            rotation, translation = transforms[surf - 1]
            bp = torch.einsum(
                "bij,bfrj->bfri", rotation, p - translation[:, None, None]
            )
            bd = torch.einsum("bij,bfrj->bfri", rotation, d)
            perpendicular = -(bp * bd).sum(-1)
            pp = bp + perpendicular[..., None] * bd
            cv = after["cv"]
            c = cv * (pp * pp).sum(-1) - 2 * pp[..., 2]
            beta = cv * (bd * pp).sum(-1) - bd[..., 2]
            disc = beta * beta - cv * c
            valid = valid & (disc > 0)
            distance = c / (before["zdir"] * torch.sqrt(disc.clamp_min(1e-30)) - beta)
            hit = pp + distance[..., None] * bd
            normal = -cv * hit + torch.tensor(
                [0.0, 0.0, 1.0], dtype=p.dtype, device=p.device
            )
            normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
            if 1 <= surf - 1 < len(self.path) - 2:
                opl = opl + before["index"] * (distance + perpendicular)
            cosine = (bd * normal).sum(-1)
            if after["mode"] == "transmit":
                ni, no = before["index"], after["index"]
                radicand = no * no - ni * ni * (1 - cosine * cosine)
                valid = valid & (radicand >= 0)
                ncos = torch.copysign(torch.sqrt(radicand.clamp_min(1e-30)), cosine)
                outgoing = (ni * bd + (ncos - ni * cosine)[..., None] * normal) / no
            elif after["mode"] == "reflect":
                outgoing = bd - 2 * cosine[..., None] * normal
            else:
                outgoing = bd
            valid = (
                valid & torch.isfinite(hit).all(-1) & torch.isfinite(outgoing).all(-1)
            )
            p, d = hit, outgoing
        return dict(hit=p, direction=d, opl=opl, valid=valid & torch.isfinite(opl))

    def metrics(self, incremental):
        traced = self.trace(incremental.to(torch.float64) + self.bias)
        hit = traced["hit"][:, :, 1:]
        direction = traced["direction"][:, :, 1:]
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True
        )
        delta = hit - traced["hit"][:, :, :1]
        projected = (delta * direction).sum(-1)
        radicand = projected * projected - (delta * delta).sum(-1) + 100.0
        sphere_opl = (
            traced["opl"][:, :, 1:] - projected - torch.sqrt(radicand.clamp_min(1e-30))
        )
        absolute = sphere_opl @ self.projector.T / self.wave
        difference = (
            (traced["opl"][:, :, 1:] - self.nominal_opl) @ self.projector.T / self.wave
        )
        abs_rms = torch.sqrt(torch.mean(absolute * absolute, -1).clamp_min(1e-20))
        diff_rms = torch.sqrt(torch.mean(difference * difference, -1).clamp_min(1e-20))
        center, edge = abs_rms[:, 0], diff_rms[:, 1:].amax(-1)
        return dict(
            center=center,
            edge=edge,
            score=torch.maximum(center, edge),
            absolute=absolute,
            difference=difference,
            valid=traced["valid"].all(-1).all(-1) & (radicand > 0).all(-1).all(-1),
        )
