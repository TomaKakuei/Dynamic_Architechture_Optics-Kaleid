"""Sequential resumable six-arm suite; no concurrent GPU trials or full3000."""
from pathlib import Path
import json
import os
import sys
import subprocess
import traceback
import msvcrt
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from durable_io import atomic_json


def stamp():
    return datetime.now(timezone.utc).isoformat()


def state(stage, **kw):
    try:
        atomic_json(ROOT/'state.json', dict(stage=stage, pid=os.getpid(), updated=stamp(), **kw))
    except PermissionError as exc:
        print(f'Telemetry deferred: {exc!r}', flush=True)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(results, complete=False):
    if not results:
        return
    first = next(iter(results.values()))
    report = dict(complete=complete, completed_arms=len(results), total_arms=6,
        baseline=first['baseline'], bounded_control=first['bounded_control'],
        arms={name: {k: r[k] for k in ('candidate', 'direct_vs_control', 'direct_vs_initial',
              'training_active_seconds', 'evaluation_active_seconds')} for name, r in results.items()},
        assistant_interpretation='pending', scope='Same reused200 selection cases; one packet, one prediction; no loop',
        automatic_full3000=False, automatic_deployment=False)
    atomic_json(ROOT/('result.json' if complete else 'partial_summary.json'), report)
    lines = ['KLA loss ablation: each new arm3epochs, same3000train and200validation.',
             'Generated metrics; assistant interpretation pending. Original Q2000>=0.45 and validity retained.',
             'Inference: one archived packet, one prediction. Selection validation reused, not an independent test.', '']
    for name, a in [('initial', report['baseline']), ('bounded_epoch3', report['bounded_control'])]:
        lines.append(f"{name}: {a['passes']}/200; WRMS {a['wrms_mean']:.8f}; invalid {a['invalid']}.")
    for name, r in results.items():
        a, paired = r['candidate'], r['direct_vs_control']
        lines.append(f"{name}: {a['passes']}/200; WRMS {a['wrms_mean']:.8f}; invalid {a['invalid']}; vs bounded +{paired['gained']}/-{paired['lost']}.")
    if complete:
        # Matched component comparisons, read saved arrays only. No inference replay.
        import numpy as np
        def pair(a, b):
            with np.load(ROOT/a/'candidate200.npz') as z:
                old = z['valid'] & (z['score'] >= .45)
            with np.load(ROOT/b/'candidate200.npz') as z:
                new = z['valid'] & (z['score'] >= .45)
                ids = z['indices']
            return dict(gained=ids[~old & new].tolist(), lost=ids[old & ~new].tolist(), net=int(new.sum()-old.sum()))
        report['component_comparisons'] = {f'{a}_to_{b}': pair(a, b) for a, b in (
            ('psf_l2', 'psf_prediction'), ('psf_l2', 'psf_q2000'),
            ('psf_prediction', 'psf_prediction_q2000'), ('psf_q2000', 'psf_prediction_q2000'))}
        atomic_json(ROOT/'result.json', report)
    (ROOT/'GENERATED_REPORT.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    if (ROOT/'result.json').exists():
        state('complete', assistant_interpretation='pending'); return
    protocol = read(ROOT/'protocol.json')
    results = {}
    for index, arm in enumerate(protocol['arms']):
        if (ROOT/'STOP_REQUESTED.json').exists():
            state('stopped', completed_arms=list(results)); return
        out = ROOT/arm
        out.mkdir(exist_ok=True)
        if (out/'result.json').exists():
            results[arm] = read(out/'result.json'); continue
        with (out/'stdout.log').open('a', encoding='utf-8') as stdout, (out/'stderr.log').open('a', encoding='utf-8') as stderr:
            child = subprocess.Popen([sys.executable, '-u', str(ROOT/'train_arm.py'), arm],
                cwd=str(ROOT), stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
            state('running', arm=arm, arm_index=index+1, total_arms=6, child_pid=child.pid,
                  completed_arms=list(results), child_state=str(out/'state.json'))
            print(json.dumps(dict(arm=arm, pid=child.pid, started=stamp())), flush=True)
            code = child.wait()
        if code != 0 or not (out/'result.json').exists():
            state('stopped' if (ROOT/'STOP_REQUESTED.json').exists() else 'failed',
                  arm=arm, child_exit_code=code, completed_arms=list(results), stderr=str(out/'stderr.log'))
            return
        results[arm] = read(out/'result.json')
        summarize(results)
    summarize(results, complete=True)
    state('complete', completed_arms=list(results), assistant_interpretation='pending', automatic_followup=False)


if __name__ == '__main__':
    with (ROOT/'suite.lock').open('a+b') as lock:
        lock.seek(0)
        if lock.read(1) == b'':
            lock.write(b'0'); lock.flush()
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            main()
        except Exception as exc:
            state('failed', error=repr(exc), traceback=traceback.format_exc()); raise
