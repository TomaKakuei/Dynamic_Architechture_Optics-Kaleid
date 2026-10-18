# One-shot API

`Kaleid(system, device="auto", checkpoint=None)` loads the selected model for
`nikon`, `trepan`, or `olympus`. `auto` selects CUDA when available, otherwise CPU.
Loading is separate from prediction; reuse one instance for successive packets.
CUDA retains the recorded precision settings: TF32 disabled for Nikon/Trepan
and enabled for Olympus. Calls restore the caller's backend settings afterward;
this package serializes its CUDA calls while those process-wide flags are set.

## Observations

| System | Single-packet shape | Values |
| --- | --- | --- |
| Nikon | `(3, 49)` | Bias-referenced wavefront residual in waves, in the calibrated field/node order. |
| Trepan | `(9, 128, 128)` | Noisy camera ADU in the trained field order. |
| Olympus Mod. | `(9, 128, 128)` | One whole-system noisy camera packet; both encoders see all nine fields. |

Add a leading batch dimension for multiple independent packets. Every value
must be finite and real. A packet is a structured set of fields, not necessarily
a single exposure. Nikon's calibrated 32 × 32 interpolation matrix is included;
camera reference images and normalization are stored with the weights.

No hidden pose, measured final error, truth label, or fitted correction gain is
accepted by `predict`. The release performs no ray tracing during inference.

## Corrections

`predict(observations)` returns a NumPy array of shape `(dofs,)` or `(batch, dofs)`.
It is an **additive normalized command**, already using the trained system's sign
and output bounds. Do not negate it again.

Coordinates repeat `[dx, dy, dz, tx, ty]` in the body order returned by
`system_info(system)`. `to_physical(command)` multiplies each coordinate by:

| System | dx, dy, dz | tx, ty | Normalized command bound |
| --- | --- | --- | --- |
| Nikon | 0.020 mm | 0.040 degrees | ±1.5 |
| Trepan | 0.100 mm | 2.5/60 degrees | ±1 |
| Olympus Mod. | 0.100 mm | 5/60 degrees | ±1 |

The angular columns are degrees, not radians. `surface_ranges` in the model
metadata identify the physical bodies in the corresponding simulation.
These scales are tied to those prescriptions. The command is a proposed motion;
the connected instrument is responsible for its coordinate frame and clearance
checks. The inference package does not actuate hardware.

## Files and custom weights

`kaleid predict SYSTEM input.npy --output result.npz` accepts `.npy` or an `.npz`
with a numeric `observations` array. The result contains `command_normalized`,
`command_physical`, `system`, `coordinates`, and `units`. Input and output must
be distinct files.

For an externally trained checkpoint, pass `checkpoint=path`. The bundle contains
`format_version=1`, `system`, `epoch`, and `state_dict`; Nikon also requires
`grid_matrix` with shape `(32, 32, 49)`. Loading uses `weights_only=True` and strict
state-dictionary matching. Calibration and normalization buffers belong to the
trained state. A checkpoint must match both the architecture and optical system.

`demo_observation(seed=17)` creates a deterministic input near stored calibration
values for interface testing. It is not an optical simulation or an alignment
accuracy benchmark.
