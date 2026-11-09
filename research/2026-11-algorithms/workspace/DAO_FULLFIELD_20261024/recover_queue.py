"""Wait for the existing queue/worker to exit, then resume saved jobs serially."""
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from durable_io import atomic_json

HERE = Path(__file__).resolve().parent


def stamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    import msvcrt
    lock = (HERE/'recovery.lock').open('a+b')
    lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        while True:
            active = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    command = ' '.join(proc.info['cmdline'] or []).replace('\\', '/')
                    if proc.info['pid'] != __import__('os').getpid() and 'python' in (proc.info['name'] or '').lower():
                        if 'DAO_FULLFIELD_20261024' in command and any(name in command for name in ('pipeline.py', 'experiment.py')):
                            active.append(proc.info['pid'])
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            if not active:
                break
            atomic_json(HERE/'recovery_state.json', dict(status='waiting_for_existing_training',
                pid=__import__('os').getpid(), waiting_for_pids=active, updated=stamp()))
            time.sleep(30)
        # Preserve the original queue's failure accounting before restarting.
        previous = json.loads((HERE/'state.json').read_text(encoding='utf-8'))
        atomic_json(HERE/'queue_before_recovery.json', previous)
        with (HERE/'recovered_pipeline.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, '-u', str(HERE/'pipeline.py')], cwd=HERE.parent,
                stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            atomic_json(HERE/'recovery_state.json', dict(status='resuming_saved_checkpoints',
                pid=__import__('os').getpid(), child_pid=child.pid, updated=stamp()))
            code = child.wait()
        atomic_json(HERE/'recovery_state.json', dict(status='queue_exited', exit_code=code,
            result_state=json.loads((HERE/'state.json').read_text(encoding='utf-8')), updated=stamp()))
    except Exception:
        atomic_json(HERE/'recovery_state.json', dict(status='failed', error=traceback.format_exc(), updated=stamp()))
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == '__main__':
    main()
