"""Serial GPU queue, restart-safe, no automatic retry of failed training."""
import csv
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from durable_io import atomic_json

HERE = Path(__file__).resolve().parent
JOBS = [('nikon', 'large', 128, 128), ('trepan', 'large', 32, 16),
        ('nikon', 'standard', 128, 128), ('trepan', 'standard', 32, 16)]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    atomic_json(path, value)


def summarize():
    rows, completed = [], []
    for system, variant, _, _ in JOBS:
        folder = HERE/f'{system}_{variant}'
        if (folder/'summary.json').exists():
            result = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            completed.append(result | dict(system=system, variant=variant))
            for label, score in [('archived_center', result['baseline']), (variant+'_fullfield', result['validation'])]:
                if any(r['system']==system and r['model']==label for r in rows):
                    continue
                rows.append(dict(system=system, model=label, count=score['count'],
                    full_field_rate=score['full_field_rate'], center07_rate=score['center07_rate'],
                    field_mean=score['field_mean'], worst_field_p90=score['worst_field_p90'],
                    invalid=score['invalid'], selected_epoch=0 if label=='archived_center' else result['selected_epoch']))
    atomic(HERE/'comparison.json', dict(rows=rows, completed_jobs=len(completed), planned_jobs=4,
        assistant_review='pending', validation_is_selection_set=True, independent_test=False, updated=stamp()))
    with (HERE/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        names = ['system', 'model', 'count', 'full_field_rate', 'center07_rate', 'field_mean', 'worst_field_p90',
                 'invalid', 'selected_epoch']
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader(); writer.writerows(rows)


def main():
    if (HERE/'TRAINING_STOPPED.json').exists():
        raise SystemExit('Training stopped by user; do not resume the queue without a new training instruction.')
    # A retained file is harmless; the OS lock is released on process exit.
    lock = (HERE/'pipeline.lock').open('a+b')
    if os.name == 'nt':
        import msvcrt
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    failures = []
    try:
        for i, (system, variant, batch, microbatch) in enumerate(JOBS):
            if (HERE/'TRAINING_STOPPED.json').exists():
                raise SystemExit('Training stopped by user.')
            folder = HERE/f'{system}_{variant}'
            if (folder/'summary.json').exists():
                continue
            preflight = json.loads((HERE/f'preflight_{system}.json').read_text(encoding='utf-8'))
            assert preflight['status']=='passed'
            command = [sys.executable, '-u', str(HERE/'experiment.py'), 'train', '--system', system,
                       '--variant', variant, '--epochs', '50', '--batch', str(batch), '--microbatch', str(microbatch)]
            logpath = HERE/f'{system}_{variant}.log'
            with logpath.open('a', encoding='utf-8') as log:
                log.write('\nRUN '+stamp()+'\n'); log.flush()
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=HERE.parent,
                                         creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                atomic(HERE/'state.json', dict(status='running', pid=os.getpid(), child_pid=child.pid,
                    job=f'{system}_{variant}', job_index=i+1, total_jobs=len(JOBS), command=command,
                    updated=stamp(), failures=failures, notification_configured=False))
                code = child.wait()
            if code:
                failures.append(dict(system=system, variant=variant, exit_code=code, log=str(logpath)))
            summarize()
        atomic(HERE/'state.json', dict(status='completed' if not failures else 'completed_with_failures',
              pid=os.getpid(), failures=failures, assistant_review='pending', updated=stamp()))
    except Exception:
        atomic(HERE/'state.json', dict(status='failed', pid=os.getpid(), error=traceback.format_exc(), updated=stamp()))
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':
    main()
