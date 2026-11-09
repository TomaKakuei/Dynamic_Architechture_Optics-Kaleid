"""Run-local configuration. No digest checks or independent replay stages."""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONUTF8'] = '1'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys, json, time, ctypes
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
ROOT = HERE.parent
PREVIOUS = ROOT / 'coordinated_7k_20261006'
BASE = ROOT / 'modular_sensorless_1k_r1'
DAO_ROOT = ROOT.parent
WORKSPACE = ROOT.parents[2]
DAO = WORKSPACE / 'Adaptive Optics' / 'ADAPTIVE_OPTICS_V15P'
BRANCHES = ('front', 'rear')
TRAIN, VALIDATION, TOTAL, EPOCHS = 7000, 500, 7500, 50
SEED = 2026101501
for _p in (ROOT, BASE / 'nonlinear_prior01', DAO_ROOT):
    sys.path.insert(0, str(_p))
import bootstrap
def current_machine_setup():
    for dependency in (bootstrap.RESTORE, bootstrap.RESTORE / 'patent_comparison', bootstrap.DAO, bootstrap.DAO / 'canonical_three_phase_universal', bootstrap.RETRAIN):
        if str(dependency) not in sys.path: sys.path.insert(0, str(dependency))
    import torch
    torch.set_num_threads(4)
bootstrap.setup = current_machine_setup
bootstrap.setup()
sys.path.insert(0, str(HERE))

def stamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def clock():
    value = ctypes.c_ulonglong()
    if os.name == 'nt' and ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)):
        return value.value / 1e7
    return time.monotonic()

def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19: raise
            time.sleep(.05 * min(attempt+1,10))

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def relocate(path):
    value = str(path).replace('\\', '/')
    if '/ECE/' in value:
        return WORKSPACE / value.split('/ECE/', 1)[1]
    return Path(path)

def state(stage, **kw):
    atomic(HERE / 'state.json', dict(stage=stage, updated=stamp(), pid=os.getpid(), **kw))




def commit(temporary, path):
    for attempt in range(20):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==19:raise
            time.sleep(.05*min(attempt+1,10))

