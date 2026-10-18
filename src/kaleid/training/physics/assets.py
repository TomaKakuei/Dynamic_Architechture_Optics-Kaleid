"""Load portable prescription and ray-launch constants, without a live tracer."""

import json
from pathlib import Path
import numpy as np
import torch

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def load_spec(name):
    def decode(value):
        if isinstance(value, dict):
            return {k: decode(v) for k, v in value.items()}
        if isinstance(value, list):
            return [decode(v) for v in value]
        if value == "Infinity":
            return float("inf")
        if value == "-Infinity":
            return -float("inf")
        return value

    return decode(json.loads((ASSETS / f"{name}.json").read_text(encoding="utf-8")))


def load_buffers(module, name, device):
    with np.load(ASSETS / f"{name}.npz", allow_pickle=False) as arrays:
        for key in arrays.files:
            module.register_buffer(
                key, torch.as_tensor(arrays[key].copy(), device=device)
            )
