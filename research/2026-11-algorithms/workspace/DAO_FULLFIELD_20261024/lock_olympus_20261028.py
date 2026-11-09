"""Preserve the user-selected Olympus revision without replay or hashes."""
from pathlib import Path
import json, shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'olympus_version_lock_20261028'
OUT.mkdir(exist_ok=True)
if (OUT / 'LOCK.json').exists():
    raise SystemExit('Lock already exists; do not overwrite it.')
folders = [
    'olympus_local_ablation150_20261027', 'olympus_secant_20261027',
    'olympus_constrained_solver_20261026', 'olympus_prediction_loop_20261026',
    'olympus_lowcost_refiner_20261026', 'olympus_repair_4ep_20261026',
    'olympus_fullpacket_20261026',
]
files = []
for name in folders:
    files.extend((ROOT / name).glob('*.py'))
source = ROOT.parent / 'RAYOPTICS_RESTORE_20260929/dao_mod03/sampling_front3_rear4/olympus_joint_20261015'
files.extend(source.glob('*.py'))
for name in folders:
    for item in ('protocol.json', 'selection.json', 'REPORT.txt', 'assistant_interpretation.json'):
        p = ROOT / name / item
        if p.exists(): files.append(p)
for rel in (
    'olympus_repair_4ep_20261026/training/last.pt',
    'olympus_repair_4ep_20261026/training/configuration.json',
    'olympus_lowcost_refiner_20261026/calibration.npz',
    'olympus_lowcost_refiner_20261026/center_gram.npy',
    'olympus_full500_20261027/result.json',
    'olympus_local_ablation150_20261027/result.json',
    'olympus_local_ablation150_20261027/predictions150.npz',
    'olympus_search64_loop2_20261028/partial_summary.json',
    'olympus_search64_loop2_20261028/STOP_REQUESTED.json',
):
    p = ROOT / rel
    if not p.exists(): raise FileNotFoundError(p)
    files.append(p)
inventory = []
for p in sorted(set(files)):
    relative = p.relative_to(ROOT.parent)
    dest = OUT / 'snapshot' / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)
    inventory.append(dict(original=str(p), snapshot=str(dest.relative_to(OUT)), bytes=p.stat().st_size))
lock = dict(
    status='locked_by_user', created=datetime.now(timezone.utc).isoformat(),
    version='Olympus_N2_refined_line_20261028', working_variant='N_line', preserved_baseline='N',
    entrypoint='olympus_local_ablation150_20261027/search.py:SearchAblation.predict_arm(prepared, N_line)',
    max_N_updates=2, response_rank=12, gains=[1., .5, .25, .125, .0625], quadratic_candidates_max=1,
    checkpoint='olympus_repair_4ep_20261026/training/last.pt', network_dtype='float32',
    score='Center ABS plus signed edge DIFF; PTTD removed; original nine-field <0.07 wave and validity rules',
    evidence=dict(original_N_full500=373, original_N_same150=112, refined_line_same150=113,
                  refined_line_full500=None, gains=[7428], losses=[], mean_quality_improvement=False),
    rejected=dict(search64_completed=97, original_N=75, refined_line=76, search64=74,
                  search64_resume=False, pattern_search=False, group_gain_changes=False),
    acquisition_packets=1, truth_in_inference=False, training_updates=0,
    automatic_resume=False, automatic_full500=False,
    snapshot_scope='Source/checkpoint/calibration/evidence archive, not a relocated standalone executable. Retain original workspace paths for cameras, archived data and response caches.',
    files=inventory)
(OUT/'LOCK.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
(OUT/'README.txt').write_text(
    'Frozen working revision: original N, two updates, refined scalar line search only.\n'
    'Original N: 373/500; paired150 N112 vs refined-line113, one marginal gain7428, no losses.\n'
    'Refined-line has NOT been evaluated on full500. Both mean WFE values are slightly worse.\n'
    'Search64 aborted at97: N75, refined-line76, search64 74. No resume.\n'
    'No fresh inference, numerical replay, hashes, training or full500 computation was performed to create this lock.\n'
    'Snapshots preserve source/weights/calibration. Original data/camera/response paths remain external dependencies.\n'
    'Restore snapshots to their recorded original layout when needed; copied entrypoints are not standalone.\n', encoding='utf-8')
handoff=ROOT/'HANDOFF.txt'
prefix=('LATEST2026-10-28: search64 ABORTED at97/150, PID30788 exited; STOP persists, no resume. '
        'N75/97,line76/97,search64 74/97. User locks originalN2+refinedline: '
        'olympus_version_lock_20261028/LOCK.json and source/weights/calibration snapshots. '
        'Line113/150 only; originalN373/500 remains full500 evidence. '
        'Next work KLA objective-alignment diagnosis; do not restart Olympus.\n\n--- Earlier snapshot retained ---\n')
handoff.write_text(prefix+handoff.read_text(encoding='utf-8-sig'),encoding='utf-8')
print(json.dumps(dict(lock=str(OUT/'LOCK.json'), files=len(inventory), bytes=sum(f['bytes'] for f in inventory))))
