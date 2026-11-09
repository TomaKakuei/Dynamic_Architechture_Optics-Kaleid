"""Small CPU integration check for the new single-field WaveFit adapter."""
from pathlib import Path
import sys
import json
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parents[1]/'Adaptive Optics/ADAPTIVE_OPTICS_V15P/kla_joint_sensorless_20261008'
sys.path.insert(0, str(SOURCE))
from common import np, clock
import torch
torch.set_num_threads(1)
from image_consistency import SelectedWaveFit

start = clock()
fit = SelectedWaveFit()
q = np.zeros(40)
all_fields = fit.t.fit(q)
row = fit.fit(q, 4)
old = all_fields[4]
diff = row['coefficients']-old['coefficients']
probe = fit.t.check_design @ np.array([diff[i, j] for i, j in fit.t.terms])
probe -= probe.mean()
report = dict(scope='Nominal CPU adapter check, no training or validation case replay',
    phase_probe_rms_difference=float(np.sqrt(np.mean(probe**2))),
    boundary_max_difference=float(abs(row['boundary']-old['boundary']).max()),
    carrier_max_difference=float(abs(row['carrier']-old['carrier']).max()),
    seconds=clock()-start)
assert report['phase_probe_rms_difference'] < .0003
assert report['boundary_max_difference'] == 0
(ROOT/'selected_field_check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
