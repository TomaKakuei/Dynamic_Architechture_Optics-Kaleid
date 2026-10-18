"""Float64 Olympus Mod. optical loss; fixed 64-grid calibration."""

import torch
from torch import nn
from .assets import load_spec, load_buffers
from .olympus_geometry import GROUPS


class TensorOptics(nn.Module):
    def __init__(self, device="cpu", center_only=False):
        super().__init__()
        spec = load_spec("olympus_optics")
        self.pupil_grid = 64
        self.wavelength_mm = spec["wavelength_mm"]
        self.n_image = spec["n_image"]
        self.path_data = spec["path_data"]
        load_buffers(self, "olympus_optics", device)
        if center_only:
            for name in (
                "launch_points",
                "launch_directions",
                "reference_centers",
                "reference_exits",
                "reference_opls",
            ):
                setattr(self, name, getattr(self, name)[:1].clone())

    def poses(self, q):
        if q.ndim != 2 or q.shape[1] != 35 or q.dtype != torch.float64:
            raise ValueError("Expected batch x35 float64 normalized coordinates")
        state = q.reshape(-1, 7, 5) * self.scales
        count = len(q)
        origins = self.origins.expand(count, -1, -1).clone()
        rotations = (
            torch.eye(3, dtype=q.dtype, device=q.device).expand(count, 25, 3, 3).clone()
        )
        zero, one = torch.zeros_like(q[:, 0]), torch.ones_like(q[:, 0])
        for group, (a, b) in enumerate(GROUPS):
            tx, ty = torch.deg2rad(state[:, group, 3]), torch.deg2rad(
                state[:, group, 4]
            )
            cx, sx, cy, sy = torch.cos(tx), torch.sin(tx), torch.cos(ty), torch.sin(ty)
            rotation = torch.stack(
                [cy, sy * sx, sy * cx, zero, cx, -sx, -sy, cy * sx, cy * cx], dim=-1
            ).reshape(count, 3, 3)
            shifted = self.origins[a : b + 1] - self.pivots[group]
            origins[:, a : b + 1] = (
                torch.einsum("bij,sj->bsi", rotation, shifted)
                + self.pivots[group]
                + state[:, group, None, :3]
            )
            rotations[:, a : b + 1] = rotation[:, None]
        return origins, rotations

    def trace(self, q):
        origins, rotations = self.poses(q)
        p = self.launch_points.expand(len(q), -1, -1, -1)
        d = self.launch_directions.expand_as(p)
        valid = torch.ones(p.shape[:-1], dtype=torch.bool, device=q.device)
        opl = torch.zeros(p.shape[:-1], dtype=q.dtype, device=q.device)
        exit_p = exit_d = None
        for surf in range(1, len(self.path_data)):
            before, after = self.path_data[surf - 1], self.path_data[surf]
            # Path index0 is prescription surface1. Preserve the reference
            # tracer's local transforms and perpendicular-plane intersection.
            ra, rb = rotations[:, surf], rotations[:, surf + 1]
            rotation = rb.transpose(-1, -2) @ ra
            translation = torch.einsum(
                "bij,bj->bi",
                ra.transpose(-1, -2),
                origins[:, surf + 1] - origins[:, surf],
            )
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
            distance = c / (
                before["z_dir"] * torch.sqrt(torch.clamp(disc, min=1e-30)) - beta
            )
            hit = pp + distance[..., None] * bd
            normal = -cv * hit + torch.tensor(
                [0.0, 0.0, 1.0], dtype=q.dtype, device=q.device
            )
            normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
            if surf - 1 < 22:
                opl = opl + before["index"] * (distance + perpendicular)
            if surf <= 22:
                for aperture in after["apertures"]:
                    ax, ay = hit[..., 0] - aperture[1], hit[..., 1] - aperture[2]
                    if aperture[0] == "Circular":
                        inside = torch.sqrt(ax * ax + ay * ay) <= aperture[3] + 1e-8
                    else:
                        inside = (abs(ax) <= aperture[3] + 1e-8) & (
                            abs(ay) <= aperture[4] + 1e-8
                        )
                    valid = valid & inside
            cosine = (bd * normal).sum(-1) / torch.linalg.vector_norm(normal, dim=-1)
            if after["mode"] == "transmit":
                ni, no = before["index"], after["index"]
                refraction = no * no - ni * ni * (1 - cosine * cosine)
                valid = valid & (refraction >= 0)
                ncos = torch.copysign(
                    torch.sqrt(torch.clamp(refraction, min=1e-30)), cosine
                )
                outgoing = (ni * bd + (ncos - ni * cosine)[..., None] * normal) / no
            elif after["mode"] == "reflect":
                outgoing = bd - 2 * cosine[..., None] * normal
            else:
                outgoing = bd
            valid = (
                valid & torch.isfinite(hit).all(-1) & torch.isfinite(outgoing).all(-1)
            )
            p, d = hit, outgoing
            if surf == 22:
                exit_p, exit_d = p, d
        valid = valid & torch.isfinite(opl)
        opl = (
            opl + (self.launch_points * self.launch_directions).sum(-1)
        ) / self.n_image
        return exit_p, exit_d, opl, valid

    def phase(self, q):
        p, d, opl, valid = self.trace(q)
        chief_valid = valid[:, :, 0]
        p, d, opl, valid = p[:, :, 1:], d[:, :, 1:], opl[:, :, 1:], valid[:, :, 1:]
        centers = self.reference_centers[None, :, None, :]
        offset = p - centers
        radius2 = ((self.reference_centers - self.reference_exits) ** 2).sum(-1)[
            None, :, None
        ]
        beta = (offset * d).sum(-1)
        c = (offset * offset).sum(-1) - radius2
        disc = beta * beta - c
        valid = valid & (disc > 0)
        root = torch.sqrt(torch.clamp(disc, min=1e-30))
        denominator = -beta + root
        safe_denominator = torch.where(
            abs(denominator) > 1e-12, denominator, torch.ones_like(denominator)
        )
        small = torch.where(
            abs(denominator) > 1e-12, c / safe_denominator, -beta - root
        )
        large = -beta + root
        distance = torch.where(abs(small) < abs(large), small, large)
        phase = (
            (opl + distance - self.reference_opls[None, :, None])
            * self.n_image
            / self.wavelength_mm
        )
        mean = torch.where(valid, phase, 0).sum(-1, keepdim=True) / valid.sum(
            -1, keepdim=True
        ).clamp_min(1)
        phase = torch.where(valid, phase - mean, 0)
        return phase, valid, chief_valid

    def high_order(self, q):
        phase, valid, chief_valid = self.phase(q)
        design = self.low_design
        weight = valid.to(phase.dtype)
        gram = torch.einsum("nr,bfn,ns->bfrs", design, weight, design)
        rhs = torch.einsum("nr,bfn->bfr", design, phase)
        coefficients = torch.linalg.solve(gram, rhs.unsqueeze(-1)).squeeze(-1)
        residual = torch.where(
            valid, phase - torch.einsum("nr,bfr->bfn", design, coefficients), 0
        )
        variance = (residual * residual).sum(-1) / weight.sum(-1)
        return variance, residual, valid, chief_valid
