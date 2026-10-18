# Kaleid · One shot

**One observation packet. One model call. One joint correction.**

Kaleid predicts coordinated alignment commands for DAO optical systems. Multiscale
spatial features and native sensor features feed attention across physical lens
groups, producing the complete correction in one forward pass.

![Kaleid one-shot architecture](docs/architecture.png)

## Included models

| System | Observation packet | Joint output | Recorded cases below 0.07 λ | Mean center WRMS |
| --- | --- | --- | --- | --- |
| Nikon | 3 × 49 wavefront samples | 35 coordinates / 7 bodies | 11 / 11 | 0.03975 λ |
| Trepan | 9 × 128 × 128 camera ADU | 40 coordinates / 8 bodies | 11 / 11 | 0.02599 λ |
| Olympus Mod. | 9 × 128 × 128 whole-system camera ADU | 35 coordinates / 7 bodies | 11 / 11 | 0.02363 λ |

These are the existing eleven-case simulation results for each optical system,
using center absolute PTTD WRMS and the original mechanical/center-pupil validity
criteria. One packet contains multiple sampled fields; it does not imply one
camera exposure. See the [model card](docs/MODEL_CARD.md) for the measurement scope.

## Install and run

Python 3.11 or newer. NumPy and PyTorch are the runtime dependencies.
The three selected checkpoints and their calibration are included in the package.

```shell
git clone https://github.com/TomaKakuei/DAO-Kaleid.git
cd DAO-Kaleid
python -m pip install .
kaleid list
kaleid demo olympus --device cpu --output outputs/demo.npz
```

The demo uses synthetic sensor-shaped input to exercise the interface. For your
own acquired packet, save a `.npy` array or a `.npz` with an `observations` array:

```shell
kaleid predict olympus observation.npy --device cpu --output outputs/correction.npz
```

```python
import numpy as np
from kaleid import Kaleid

aligner = Kaleid("olympus", device="auto")
observation = np.load("observation.npy", allow_pickle=False)  # (9, 128, 128)
command = aligner.predict(observation)                       # (35,)
physical_command = aligner.to_physical(command)               # mm and degrees
```

Each model consumes the measured packet and returns an additive correction.
There is no online optical optimization, recapture loop, or ground-truth input.
CPU and CUDA use the same selected weights. Instrument controllers apply their
own coordinate mapping and mechanical constraints before actuation.

## Architecture and integration

- **Nikon:** three spatial encoders, 147 native sensor nodes, and joint module attention.
- **Trepan:** a spatial encoder and raw camera patches condition eight body tokens.
- **Olympus Mod.:** both branch encoders receive the same whole-system packet;
  joint attention coordinates all seven bodies.

The repository contains the current inference code, three checkpoints, API docs,
an architecture figure, a small example, and portable tests. Each optical system
has its own trained model and calibration. New prescriptions require matched
sensor calibration and training.

[Input/output contract](docs/API.md) · [Model card](docs/MODEL_CARD.md) ·
[Architecture PDF](docs/architecture.pdf) · [Editable SVG](docs/architecture.svg)

```shell
python examples/one_shot.py trepan
python -m unittest discover -s tests -v
```
