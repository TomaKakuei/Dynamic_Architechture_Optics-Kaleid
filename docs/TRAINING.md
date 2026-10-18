# Training Kaleid

Kaleid learns a joint mechanical correction from one whole-system observation.
This package exposes the trainable architecture and physical objectives of the
current Nikon, Trepan and Olympus Mod. one-shot models. Training is offline;
inference remains one network evaluation.

## Data

Create a manifest with explicit training and validation partitions. Paths are
relative to the manifest; no automatic split mixes the two partitions.

```json
{
  "system": "trepan",
  "train": [{"path": "train/batch_0000.npz"}],
  "validation": [{"path": "validation/batch_0000.npz"}]
}
```

Each NPZ contains two real, finite arrays:

| Array | Nikon | Trepan | Olympus Mod. |
| --- | --- | --- | --- |
| `observations` | N × 3 × 49, wavefront residual in waves | N × 9 × 128 × 128, camera ADU | N × 9 × 128 × 128, whole-system camera ADU |
| `targets` | N × 35 | N × 40 | N × 35 |

`targets` are the **initial normalized pose errors**, not desired commands.
The corrected pose is `targets + command`. Coordinates are ordered body by body
as `dx, dy, dz, tx, ty`; scales and surface ranges are in
[`systems.json`](../src/kaleid/systems.json). Use the matching observation model,
field order and calibration described in the [API](API.md).

The adapter also accepts the original dataset keys: Nikon `raw`, `maps`, `targets`;
camera systems `speckles`, with `q` for Trepan or `targets_normalized` for Olympus.
Optional Nikon maps must be
N × 3 × 32 × 32 and come from the same native packet. Otherwise the fixed
interpolation matrix constructs them. An entry can specify `indices: [0, 2, 4]`
to select rows. Duplicate rows and direct file/row overlap across splits are
rejected; keep scene/seed identities disjoint across separately saved files too.

For the Olympus joint/prior schedule, add training records with `kind: "front"`
and `kind: "rear"`, alongside `kind: "joint"` (the default). All three cohorts
must have the same size. Each epoch interleaves joint batches with front-prior
batches on odd epochs and rear-prior batches on even epochs. Every input remains
a whole-system image stack and every target contains all 35 coordinates.
Validation uses joint observations only. Omitting both priors trains on the joint
cohort alone; it does not recreate the original mixed schedule.

The loader shuffles files and rows deterministically and caches three NPZ shards.
Use reasonably sized shards (for example, 25 camera packets) to bound memory and
avoid repeatedly decompressing a large file. It does not load an entire camera
dataset into GPU memory.

## Calibration and initialization

The default starts with **random learned parameters**. It loads only fixed sensor
references, coordinate encodings, normalization statistics and interpolation
constants from `training/assets/*_calibration.npz`. These files contain no learned
network weights. The supplied calibration belongs to the included prescriptions
and original training observations.

For a new matched dataset, supply `--calibration calibration.npz` with these keys:

| System | Required calibration arrays |
| --- | --- |
| Nikon | `raw_mean`, `raw_std` (3 × 49); `target_mean`, `target_std` (35); `node_coordinates` (147 × 4); `grid_matrix` (32 × 32 × 49) |
| Trepan | `reference` (9 × 128 × 128); `coordinates` (2304 × 4) |
| Olympus Mod. | `front.` and `rear.` prefixes for `reference`, `noise_std` (9 × 128 × 128), `target_mean`, `target_std` (15 front / 20 rear) |

Estimate normalization statistics on training data only. Sensor references must
match the acquisition model, and scale arrays must be positive. A different
prescription also needs a corresponding optical loss and mechanical model;
changing normalization alone does not adapt the ray tracers to a new lens.

Use `--init pretrained` to fine-tune the example model, or `--init path.pt` for
another compatible Kaleid deployment bundle. A bundle includes its calibration;
`--calibration` is for random initialization. Gradients flow through the underlying
PyTorch model and physical objective, without calling inference-only `Kaleid.predict`.

## Physical objectives

Let `q` be the initial pose, `c` the bounded command, and `w` the center absolute
PTTD wavefront RMS in wavelengths at `q + c`. PTTD removes piston, tip, tilt and
defocus. Optical propagation uses float64; networks use float32. Each objective
retains its system's ray and command conventions.

| System | Optical term | Auxiliary terms and validity |
| --- | --- | --- |
| Nikon | `sqrt(1 + (w/0.07)^2) - 1 + 0.5*relu(w/0.07 - 1)^2` | 0.02 × standardized pose SmoothL1 (beta 0.25); invalid ray endpoints contribute 100. |
| Trepan | `ell/(1+ell)`, `ell = log(1 + (w/0.07)^2)` | 0.05 × command SmoothL1 + 10 × smooth gap penalty; pupil support at least 98% and certified straight command path. Invalid endpoints have a penalty above the valid optical range. |
| Olympus Mod. | Same bounded optical term as Trepan | 0.05 × command SmoothL1 + 10 × pose-domain penalty; invalid states contribute 2; chief-ray validity, pupil support at least 20%, and certified straight command path. |

Trepan and Olympus include prescription-derived lens-body clearance models,
including nested menisci. These describe the modeled lens bodies, not unknown
mounts or actuators. Nikon retains its original ray-validity objective; it does
not add a new collision certificate. Visibility and geometry certificates are
discrete decisions; gradients pass through the continuous optical and auxiliary terms.

Nikon's off-axis bias-referenced difference metric remains available in its tracer;
it is not substituted for the center absolute metric. Trepan uses the 65-grid
pupil trace; Olympus uses its fixed 64-grid launch calibration. Camera-system
validation traces all nine fields and reports the center criterion, validity
and field means. Invalid cases remain in the denominator.

## Recipes, saving and resuming

All recipes use AdamW, weight decay 1e-4, gradient clipping at norm 5,
50 epochs by default, and cosine learning-rate decay.

| System | Batch size | Existing encoders / added coordination LR | Minimum LR | Full validation |
| --- | --- | --- | --- | --- |
| Nikon | 128 | 2e-5 / 1e-4 | 2e-6 | Every epoch |
| Trepan | 32 | 3e-4 / 3e-4 | 3e-6 | Every epoch |
| Olympus Mod. | 25 | 2e-5 / 1e-4 | 2e-6 | Every 5 epochs and final epoch |

Epoch zero is evaluated and retained as a candidate. Selection maximizes valid
validation cases with center WRMS < 0.07 waves, then minimizes center P90 and mean.
Olympus keeps branch BatchNorm statistics frozen as in the source training.
Nikon and Trepan disable TF32; Olympus retains its source TF32 setting on CUDA.

```shell
kaleid-train olympus --data data/olympus/manifest.json --output runs/olympus --device cuda --micro-batch 5
kaleid-train olympus --data data/olympus/manifest.json --output runs/olympus --device cuda --micro-batch 5 --resume
```

`--micro-batch` accumulates gradients before an update. With trainable BatchNorm
(Nikon/Trepan), changing it also changes batch statistics: it is a memory option,
not a promise of identical trajectories. Choose it before starting a run.
`--max-steps N` pauses after N additional optimizer updates; omit it for the full
schedule. `--save-every` controls checkpoint frequency. An interrupted batch is
replayed from the last saved state. Output directories are not silently overwritten.

| Output | Purpose |
| --- | --- |
| `best.pt` | Selected deployment bundle, directly loadable by `Kaleid` |
| `last.pt` | Most recently exported model |
| `resume.pt` | Model/calibration, optimizer, RNG, next epoch/batch, best score and accumulated loss |
| `history.json` | Epoch losses and validation summaries |

Resume requires unchanged data, recipe, epoch budget, batch size and seed. The
configuration is checked, but dataset contents are not hashed. CPU resume is
tested against uninterrupted updates. CUDA is not promised to be bitwise
deterministic across kernels or devices.

## Relationship to the existing models

The included example weights came from the existing training runs:

| System | Training / validation observations | Initialization history |
| --- | --- | --- |
| Nikon | 24,000 / 6,000 | Earlier fitted encoders, followed by joint center-loss training |
| Trepan | 14,000 / 1,000 | Random parameters; training-set camera reference |
| Olympus Mod. | 7,000 joint plus 7,000 alternating prior observations per epoch / 500 joint validation | Earlier fitted front/rear encoders, a retained startup update, then joint training |

This release includes the current models, optical kernels, objectives, calibration
and portable training loop. It does **not** include the full rendered datasets,
earlier encoder-training stages or historical optimizer states. The portable
loader uses file-local shuffling; the original in-memory runs used their own
sampling order. The runner omits the old partial validation/probe reports between
full validations. Starting from random parameters is a supported new training
run, not a bit-for-bit recreation of the three checkpoints. The retained startup
update is not silently applied to a fresh run.

Supply or generate paired observations using the matching acquisition model to
train for accuracy. The `--smoke` generator is an interface fixture and cannot
replace that dataset. Local experiment logs and full datasets are not included
in the Git repository.
