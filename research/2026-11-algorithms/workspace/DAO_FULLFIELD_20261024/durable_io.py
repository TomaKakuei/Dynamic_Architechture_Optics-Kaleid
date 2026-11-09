"""Atomic output tolerant of transient Windows sharing/rename failures."""
import json
import os
from pathlib import Path
import time
import uuid


def retry_io(operation, timeout=15.):
    deadline = time.monotonic()+timeout
    delay = .025
    while True:
        try:
            return operation()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay*2, .5)


def temporary(path):
    return path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temporary(path)
    text = json.dumps(value, indent=2, allow_nan=False)
    retry_io(lambda: tmp.write_text(text, encoding='utf-8'))
    retry_io(lambda: os.replace(tmp, path))


def save_torch(path, value):
    import torch
    path = Path(path)
    tmp = temporary(path)
    retry_io(lambda: torch.save(value, tmp))
    retry_io(lambda: os.replace(tmp, path))
