"""Tiny interface exercise; these packets are not rendered optical training data."""

from pathlib import Path
import json
import numpy as np
from .physics.assets import ASSETS
from ..predictor import system_info


def make_smoke_data(system, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    info = system_info(system)
    with np.load(ASSETS / f"{system}_calibration.npz", allow_pickle=False) as cal:
        reference = (
            cal["raw_mean"]
            if system == "nikon"
            else cal["reference" if system == "trepan" else "front.reference"]
        )
    rng = np.random.default_rng(1729)
    for split in ("train", "validation"):
        observations = reference + rng.normal(
            0, 0.001 if system == "nikon" else 1.0, (4, *info["input_shape"])
        )
        targets = rng.uniform(-0.02, 0.02, (4, info["dofs"]))
        path = directory / f"{split}.npz"
        if not path.exists():
            np.savez_compressed(
                path, observations=observations.astype(np.float32), targets=targets
            )
    manifest = directory / "manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                dict(system=system, train=["train.npz"], validation=["validation.npz"]),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return manifest
