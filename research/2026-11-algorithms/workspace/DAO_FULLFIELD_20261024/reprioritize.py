"""Preserve a stable checkpoint before replacing the old queue priority."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import psutil
import torch
from durable_io import atomic_json

HERE = Path(__file__).resolve().parent


def checked(pid, script):
    process = psutil.Process(pid)
    command = ' '.join(process.cmdline()).replace('\\', '/')
    assert 'DAO_FULLFIELD_20261024' in command and script in command, command
    return process


def stop(process):
    process.terminate()
    process.wait(timeout=10)


def main():
    queue = json.loads((HERE/'state.json').read_text(encoding='utf-8'))
    repair = json.loads((HERE/'recovery_state.json').read_text(encoding='utf-8'))
    parent = checked(queue['pid'], 'pipeline.py')
    worker = checked(queue['child_pid'], 'experiment.py')
    coordinator = checked(repair['pid'], 'recover_queue.py')
    assert queue['job']=='trepan_standard'
    stop(coordinator)
    worker.suspend()
    stopped = False
    try:
        path = HERE/'trepan_standard/resume.pt'
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['optimizer']['state'] and 'rng' in saved and 'cuda_rng' in saved
        assert saved['config']['system']=='trepan' and saved['config']['variant']=='standard'
        evidence = HERE/'priority_change_20261024'
        evidence.mkdir(exist_ok=True)
        shutil.copy2(path, evidence/'trepan_standard_resume.pt')
        atomic_json(evidence/'checkpoint.json', dict(epoch=saved['epoch'], next_batch=saved['next_batch'],
            processed=saved['processed'], last_completed_epoch=saved['history'][-1]['epoch'],
            reason='User prioritizes both large networks', old_queue=queue, old_recovery=repair,
            timestamp=datetime.now(timezone.utc).isoformat()))
        stop(parent)
        stop(worker)
        stopped = True
        atomic_json(HERE/'trepan_standard/state.json', dict(status='queued_after_large_networks',
            epoch=saved['epoch'], batch=saved['next_batch'], checkpoint_saved=True,
            reason='User priority change', updated=datetime.now(timezone.utc).isoformat()))
        atomic_json(HERE/'recovery_state.json', dict(status='superseded_by_large_first_queue',
            previous_pid=repair['pid'], updated=datetime.now(timezone.utc).isoformat()))
        atomic_json(HERE/'state.json', dict(status='preparing_large_first_queue',
            order=['nikon_large','trepan_large','nikon_standard','trepan_standard'],
            updated=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(dict(saved_epoch=saved['epoch'], saved_next_batch=saved['next_batch'],
                             stopped_pids=[coordinator.pid,parent.pid,worker.pid])))
    finally:
        if not stopped and worker.is_running():
            worker.resume()


if __name__=='__main__':
    main()
