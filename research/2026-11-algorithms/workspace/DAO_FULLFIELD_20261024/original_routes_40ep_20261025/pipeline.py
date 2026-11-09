"""Nikon-only queue, corrected user scope; restartable training checkpoints."""
from pathlib import Path
import ctypes
import json
import msvcrt
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from durable_io import atomic_json


def stamp():return datetime.now(timezone.utc).isoformat()


def main():
    lock=(HERE/'pipeline.lock').open('a+b')
    lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    stages=['generate_nikon','train_nikon']
    completed=[]
    atomic_json(HERE/'started.json',dict(pid=os.getpid(),python=sys.executable,started=stamp(),stages=stages))
    try:
        for stage in stages:
            if (HERE/'STOP').exists():
                raise RuntimeError('Local STOP requested; checkpoints retained')
            with (HERE/f'{stage}.log').open('a',encoding='utf-8',buffering=1) as log:
                child=subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py'),stage],
                    cwd=str(HERE),stdout=log,stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                atomic_json(HERE/'state.json',dict(status='running',pid=os.getpid(),child_pid=child.pid,
                    stage=stage,completed=completed,updated=stamp()))
                code=child.wait()
                if code:raise RuntimeError(f'{stage} exited {code}; inspect {stage}.log')
            completed.append(stage)
        results={name:json.loads((HERE/name/'summary.json').read_text()) for name in ('nikon',)}
        atomic_json(HERE/'results.json',dict(status='completed',assistant_review='pending',systems=results,updated=stamp()))
        atomic_json(HERE/'state.json',dict(status='completed',pid=os.getpid(),completed=completed,updated=stamp()))
    except Exception:
        atomic_json(HERE/'state.json',dict(status='failed',pid=os.getpid(),completed=completed,
            error=traceback.format_exc(),updated=stamp()))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':main()
