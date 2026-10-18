"""Trainable model construction; pretrained parameters are optional."""

from pathlib import Path
import numpy as np
import torch
from .physics.assets import ASSETS


def build_model(system, *, calibration=None, init=None, device="cpu"):
    from ..predictor import system_info

    system_info(system)
    if init is not None and calibration is not None:
        raise ValueError(
            "An initialization bundle supplies its calibration; omit calibration"
        )
    if init is not None:
        from .. import Kaleid

        loaded = Kaleid(
            system,
            checkpoint=None if str(init) == "pretrained" else init,
            device=device,
        )
        return loaded._model.train(), loaded._grid
    path = Path(calibration) if calibration else ASSETS / f"{system}_calibration.npz"
    with np.load(path, allow_pickle=False) as source:
        cal = {key: torch.from_numpy(source[key].copy()) for key in source.files}
    grid = None
    if system == "nikon":
        from ..models.nikon import NikonModel

        model = NikonModel()
        state = model.state_dict()
        for key in (
            "raw_mean",
            "raw_std",
            "target_mean",
            "target_std",
            "node_coordinates",
        ):
            state[key] = cal[key]
        model.load_state_dict(state, strict=True)
        grid = cal["grid_matrix"].numpy()
        if grid.shape != (32, 32, 49) or not np.isfinite(grid).all():
            raise ValueError("Invalid Nikon interpolation matrix")
    elif system == "trepan":
        from ..models.trepan import TrepanModel

        model = TrepanModel(cal["reference"])
        model.coordinates.copy_(cal["coordinates"])
    else:
        from ..models.olympus import OlympusModel

        model = OlympusModel(
            {
                branch: {
                    k: cal[f"{branch}.{k}"]
                    for k in ("target_mean", "target_std", "reference", "noise_std")
                }
                for branch in ("front", "rear")
            }
        )
    for key, value in model.named_buffers():
        if not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite calibration: {key}")
        if (
            key.endswith(("raw_std", "target_std", "noise_std"))
            and not (value > 0).all()
        ):
            raise ValueError(f"Calibration scales must be positive: {key}")
    return model.to(device).train(), grid


def export_bundle(system, model, grid, epoch):
    bundle = dict(
        format_version=1,
        system=system,
        epoch=int(epoch),
        state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
    )
    if system == "nikon":
        bundle["grid_matrix"] = torch.as_tensor(grid).clone()
    return bundle


def forward(system, model, observations, grid=None, maps=None):
    if system == "nikon":
        if maps is None:
            # Keep the original NumPy interpolation/cast order used by inference.
            raw = observations.detach().cpu().numpy()
            maps = torch.as_tensor(
                np.einsum("bfk,hwk->bfhw", raw, grid).astype(np.float32),
                device=observations.device,
            )
        prediction = model(maps, observations)
        command = -(prediction * model.target_std + model.target_mean).clamp(-1.5, 1.5)
        return command, prediction
    return model.command(observations), None
