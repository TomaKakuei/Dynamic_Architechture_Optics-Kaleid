# Kaleid one-shot models

Kaleid is the alignment algorithm; DAO is the optical system architecture.
This release provides three system-specific, trained one-shot models.

## Released checkpoints

| System | Selected epoch | Structure | Applied command |
| --- | --- | --- | --- |
| Nikon | 49 | Three multiscale encoders, native-node attention, module attention | Negative pose estimate, clipped to ±1.5 |
| Trepan | 49 | Multiscale encoder, patch attention, eight-body attention | Negative tanh of the joint output |
| Olympus Mod. | 45 | Two branch encoders of the same whole-system packet, seven-body attention | Negative pose estimate, clipped to ±1 |

The checkpoints include all inference parameters, reference values, and
normalization buffers. Optimizer states, training datasets, local paths, logs,
and development metadata are not distributed. The network classes remain
ordinary PyTorch modules; this package is a deployment release, not a full
training-data or optical-simulator release.

## Recorded simulation results

Each row refers to the original eleven registered cases for that system.
The primary criterion was center-field absolute PTTD wavefront RMS below
0.07 wavelengths, with the original mechanical and center-pupil validity checks.

| System | Passed | Mean center WRMS, waves | Initially below target |
| --- | --- | --- | --- |
| Nikon | 11 / 11 | 0.039750931 | 0 |
| Trepan | 11 / 11 | 0.025992566 | 0 |
| Olympus Mod. | 11 / 11 | 0.023632833 | 2 |

Each route used one observation packet, one joint prediction, and one correction.
Nikon and Trepan use the recorded October 14 results; Olympus uses the joint
epoch-45 model recorded on October 15, 2026 (Pacific time). These fixed cases are
separate from training validation. No new optical benchmark or training run was
performed to prepare this deployment package.

The primary metric is the center field; it does not establish full-field
diffraction-limited performance. These are simulated systems with matched
prescriptions and sensor models, not a demonstrated hardware alignment result.
The registered eleven cases also do not establish accuracy on arbitrary new
prescriptions. Recalibration and system-matched training are required there.

## Inference boundary

The observation packet is the only inference input. Differentiable optics was
used in training; it is outside the online inference path. The package performs
no feedback acquisition, numerical refinement, or truth-based routing. A single
packet can contain multiple fields and must not be interpreted as necessarily
one camera exposure.

The architecture diagram shows the online path and the offline training path.
Its lens drawings are schematic. CPU tests check the packaged interfaces and
model execution; they do not rerun the physical eleven-case benchmarks.
