"""Audit existing KLA propagation direction without editing or running a model job."""
from pathlib import Path
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / 'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
ENV = Path('C:/Users/arthu/anaconda3/envs/nn_310')
sys.path.append(str(ENV / 'Lib/site-packages'))
_dll = os.add_dll_directory(str(ENV / 'Library/bin'))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
sys.path.insert(0, str(SOURCE))

import numpy as np
from optics import numpy_trace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Independently transcribed numeric rows from US6842298B1 Table 1.
# Row zero is STO; rows 1..30 are patent surfaces 2..31.
table1 = [
    (0., 15.188841), (-81.627908, 3.5), (-18.040685, 22.449116),
    (18.74457, 2.), (795.137592, 1.998104), (84.996662, 5.),
    (40.302422, 97.532362), (-78.567476, 5.), (-132.110046, -5.),
    (-78.567476, -97.532362), (40.302422, -5.), (84.996662, 5.),
    (40.302422, 97.532362), (-78.567476, 5.), (-132.110046, 14.180612),
    (41.906043, 2.999944), (-19.645329, .499948), (10.206534, 6.643053),
    (6.314274, 5.385248), (-6.571777, 8.442713), (-11.608676, 19.085531),
    (29.380754, 2.999908), (25.288697, 4.186877), (55.554188, 6.84081),
    (-51.735654, .5), (53.425082, 5.141563), (-275.827116, .5),
    (27.209707, 5.295973), (85.400041, .5), (13.757522, 6.782701),
    (69.464423, 8.236734),
]
paths = [SOURCE / n for n in ('prescription.json', 'optics.py', 'camera.py', 'protocol.json')]
before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
spec = json.loads(paths[0].read_text(encoding='utf-8'))
actual = np.array([[s['r'], s['t']] for s in spec['surfaces']])
expected = np.array(table1)
assert actual.shape == expected.shape
assert np.array_equal(actual, expected)

# Tiny CPU ray fan only. No diffraction rendering, inference, training, or GPU work.
xy = np.array([[0., 0.], [0., -.25], [0., .25], [0., -.75], [0., .75], [0., -1.], [0., 1.]])
rays = []
for fi in (0, 4, 8):
    p, d, opl, valid = numpy_trace(xy, spec['angles_rad'][fi])
    image_z = spec['image_z']
    endpoint = p + ((image_z - p[:, 2]) / d[:, 2])[:, None] * d
    rays.append(dict(field_index=fi, initial_z_mm=0.,
        initial_direction=[0., float(np.sin(spec['angles_rad'][fi])), float(np.cos(spec['angles_rad'][fi]))],
        last_surface_xyz_mm=p.tolist(), last_direction=d.tolist(),
        at_registered_image_plane_xyz_mm=endpoint.tolist(), valid=valid.tolist(),
        chief_y_at_registered_plane_mm=float(endpoint[0, 1]),
        registered_reference_xyz_mm=spec['reference_points_mm'][fi]))
    assert valid.all()
    assert np.max(abs(endpoint[0] - np.array(spec['reference_points_mm'][fi]))) < 1e-10
    assert (d[:, 2] > 0).all()

after = {str(p.relative_to(ROOT)): sha(p) for p in paths}
assert before == after
result = dict(
    audit_date='2026-10-29 America/Los_Angeles',
    user_intended_direction='sample -> collection objective -> image-forming optics -> detector',
    actual_current_direction='plane waves at external pupil z=0 -> objective -> high-NA sample-side reference',
    current_recent_run_entry='DAO_FULLFIELD_20261024/kla_feedback_loop150_20261028/run.py',
    prescription_source=spec['source'],
    table1_numeric_rows_checked=len(table1),
    radius_thickness_max_abs_difference=float(abs(actual - expected).max()),
    actual_final_surface_z_mm=spec['surfaces'][-1]['z'],
    registered_image_z_mm=spec['image_z'],
    registered_image_plane_physical_role='sample plane when objective is used for detection',
    current_camera='ideal pupil-coordinate relay -> fixed diffuser -> 2 mm propagation -> intensity sensor',
    current_camera_source_scope='Simulated calibration sensor; camera.py explicitly says not patent hardware.',
    detector_side_image_forming_prescription_present=False,
    tdi_sample_image_model_present=False,
    reverse_trace_equivalence_to_requested_complete_detector_system='Not established. Reversibility alone does not validate sensor conjugates, detector relay, or measured inputs.',
    no_claim='This audit does not establish that all historical objective-level results are wrong or establish corrected detector-level performance.',
    calls=dict(numpy_ray_fans=3, rays_per_fan=7, diffraction_renders=0, network_calls=0, training_updates=0),
    changed_production_sources=False,
    source_hashes=after,
    ray_fans=rays,
)
(HERE / 'audit_result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ('ray_fans', 'source_hashes')}, indent=2))
