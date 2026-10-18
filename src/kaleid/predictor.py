"""One observation packet to one bounded, joint correction."""
from copy import deepcopy
from contextlib import contextmanager
import json
from pathlib import Path
import threading

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent
_SYSTEMS = json.loads((_ROOT / "systems.json").read_text(encoding="utf-8"))
_CUDA_PRECISION_LOCK = threading.RLock()


@contextmanager
def _precision(device, system):
    # Preserve each recorded runner's arithmetic and restore process settings.
    # Serialize this package's CUDA calls because PyTorch backend flags are global.
    if device.type != "cuda":
        yield
        return
    with _CUDA_PRECISION_LOCK:
        previous = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        allow_tf32 = system == "olympus"
        try:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
            torch.backends.cudnn.allow_tf32 = previous_cudnn


def available_systems():
    return tuple(_SYSTEMS)


def system_info(system):
    if system not in _SYSTEMS:
        raise ValueError(f"Unknown system {system!r}; choose {', '.join(_SYSTEMS)}")
    return deepcopy(_SYSTEMS[system])


class Kaleid:
    """Load one system's trained model and predict additive actuator commands.

    Inputs are observations only. Output coordinates are normalized and ordered
    body by body as dx, dy, dz, tx, ty. Use ``to_physical`` for mm/degrees.
    """

    def __init__(self, system, *, device="auto", checkpoint=None):
        self.info = system_info(system)
        self.system = system
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device='cpu'")
        path = Path(checkpoint) if checkpoint is not None else _ROOT / "weights" / f"{system}.pt"
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if bundle.get("format_version") != 1 or bundle.get("system") != system:
            raise ValueError("Checkpoint format or optical system does not match")
        state = bundle["state_dict"]
        self._grid = None
        if system == "nikon":
            from .models.nikon import NikonModel
            model = NikonModel()
            self._grid = bundle["grid_matrix"].numpy()
            if self._grid.shape != (32, 32, 49) or not np.isfinite(self._grid).all():
                raise ValueError("Invalid Nikon sensor-grid calibration")
        elif system == "trepan":
            from .models.trepan import TrepanModel
            model = TrepanModel(state["reference"])
        else:
            from .models.olympus import OlympusModel
            calibration = {
                branch: {key: state[f"{branch}.{key}"] for key in
                         ("target_mean", "target_std", "reference", "noise_std")}
                for branch in ("front", "rear")
            }
            model = OlympusModel(calibration)
        model.load_state_dict(state, strict=True)
        self._model = model.to(self.device).eval()
        self.epoch = bundle["epoch"]

    def _observations(self, observations):
        if isinstance(observations, torch.Tensor):
            observations = observations.detach().cpu().numpy()
        array = np.asarray(observations)
        shape = tuple(self.info["input_shape"])
        single = array.shape == shape
        if single:
            array = array[None]
        if array.ndim != len(shape) + 1 or array.shape[1:] != shape or len(array) == 0:
            raise ValueError(f"Expected {shape} or (batch, {', '.join(map(str, shape))})")
        if (not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array)
                or not np.isfinite(array).all()):
            raise ValueError("Observations must be finite real numbers")
        with np.errstate(over="ignore", invalid="ignore"):
            numeric = array.astype(np.float32)
        if not np.isfinite(numeric).all():
            raise ValueError("Observations exceed the float32 input range")
        return array, numeric, single

    @torch.inference_mode()
    def predict(self, observations):
        """Return one normalized additive correction per input packet."""
        original, numeric, single = self._observations(observations)
        raw = torch.as_tensor(numeric, device=self.device)
        with _precision(self.device, self.system), torch.autocast(self.device.type, enabled=False):
            if self.system == "nikon":
                # Preserve the registered NumPy interpolation and its cast order.
                maps = np.einsum('bfk,hwk->bfhw', original, self._grid).astype(np.float32)
                image = torch.as_tensor(maps, device=self.device)
                command = -self._model.physical_prediction(image, raw).clamp(-1.5, 1.5)
            else:
                command = self._model.command(raw)
            output = command.cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("Model produced a non-finite command")
        return output[0] if single else output

    __call__ = predict

    def to_physical(self, command):
        """Scale normalized coordinates to mm, mm, mm, degrees, degrees."""
        values = np.asarray(command)
        if (values.ndim not in (1, 2) or values.shape[-1] != self.info["dofs"]
                or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
                or not np.isfinite(values).all()):
            raise ValueError("Expected finite real command coordinates for this system")
        return values * np.tile(self.info["scales"], self.info["dofs"] // 5)

    def demo_observation(self, seed=17):
        """Synthetic input for interface smoke tests, not a simulated optical scene."""
        rng = np.random.default_rng(seed)
        if self.system == "nikon":
            base = self._model.raw_mean.detach().cpu().numpy()
            return base + rng.normal(0., .015, base.shape)
        reference = (self._model.reference if self.system == "trepan"
                     else self._model.front.reference).detach().cpu().numpy()
        return np.clip(reference + rng.normal(0., 2., reference.shape), 0, 65535).astype(np.float32)
