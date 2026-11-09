"""Bounded CPU-only reproduction of the Windows rename failure and recovery."""
import ctypes
import json
from pathlib import Path
import shutil
import threading
import time
import torch
from durable_io import atomic_json, save_torch

HERE = Path(__file__).resolve().parent
OUT = HERE/'recovery_evidence'
OUT.mkdir(exist_ok=True)
kernel = ctypes.windll.kernel32
kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                               ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
kernel.CreateFileW.restype = ctypes.c_void_p
kernel.CloseHandle.argtypes = [ctypes.c_void_p]


def hold(path):
    # Permit readers/writers but not delete/rename, as a transient file reader can.
    handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
    assert handle not in (None, ctypes.c_void_p(-1).value)
    return handle


def main():
    for path in HERE.glob('failure_*_train.json'):
        destination = OUT/path.name
        if not destination.exists():
            shutil.copy2(path, destination)
    path = OUT/'locked_status.json'
    atomic_json(path, {'version': 0})
    replacement = OUT/'direct_replace.json'
    replacement.write_text('{"version":1}', encoding='utf-8')
    handle = hold(path)
    try:
        replacement.replace(path)
        raise AssertionError('Expected Windows sharing failure')
    except PermissionError as error:
        original = repr(error)
    timer = threading.Timer(.3, lambda: kernel.CloseHandle(handle))
    timer.start()
    tick = time.monotonic()
    atomic_json(path, {'version': 2})
    seconds = time.monotonic()-tick
    timer.join()
    assert json.loads(path.read_text()) == {'version': 2}
    checkpoint = OUT/'locked_checkpoint.pt'
    save_torch(checkpoint, {'value': torch.tensor([1., 2.])})
    handle = hold(checkpoint)
    timer = threading.Timer(.3, lambda: kernel.CloseHandle(handle))
    timer.start()
    save_torch(checkpoint, {'value': torch.tensor([3., 4.])})
    timer.join()
    assert torch.equal(torch.load(checkpoint, weights_only=False)['value'], torch.tensor([3., 4.]))
    resumes = []
    for job in ('nikon_large', 'nikon_standard', 'trepan_large'):
        saved = torch.load(HERE/job/'resume.pt', map_location='cpu', weights_only=False)
        resumes.append(dict(job=job, next_epoch=saved['epoch'], next_batch=saved['next_batch'],
            completed_epochs=max(h['epoch'] for h in saved['history']),
            optimizer_present=bool(saved['optimizer']['state']), rng_present='rng' in saved and 'cuda_rng' in saved))
        del saved
    result = dict(status='passed', original_error_reproduced=original, json_retry_seconds=seconds,
                  torch_checkpoint_retry_passed=True, saved_resume_points=resumes, training_updates=0)
    atomic_json(OUT/'io_recovery_check.json', result)
    print(json.dumps(result, indent=2))


if __name__=='__main__':
    main()
