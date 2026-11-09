"""Wait on the existing process handle, then conditionally debug/train serially."""
from pathlib import Path
import os,sys,json,ctypes,subprocess,traceback,msvcrt
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from durable_io import atomic_json


def stamp():return datetime.now(timezone.utc).isoformat()


def state(status,**kw):
    atomic_json(HERE/'continuation_state.json',dict(status=status,pid=os.getpid(),updated=stamp(),**kw))


def run_child(label,args):
    if (HERE/'STOP').exists():
        state('paused',stage=label);return False
    with (HERE/f'{label}.log').open('a',encoding='utf-8') as log:
        child=subprocess.Popen([sys.executable,'-u',*args],cwd=str(HERE),stdout=log,stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        state('running',stage=label,child_pid=child.pid)
        code=child.wait()
    if code:raise RuntimeError(f'{label} failed with exit{code}; inspect {label}.log')
    return True


def main():
    lock=(HERE/'continuation.lock').open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
    kernel.WaitForSingleObject.restype=ctypes.c_uint32
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=None if (HERE/'fixed_bn_4ep/summary.json').exists() else kernel.OpenProcess(0x100000,False,30628)
    try:
        if handle:
            state('waiting_existing_run',waiting_on_pid=30628,process_handle_opened=True)
            while True:
                if (HERE/'STOP').exists():state('paused',waiting_on_pid=30628);return
                result=kernel.WaitForSingleObject(handle,5000)
                if result==0:break
                if result!=258:raise RuntimeError(f'WaitForSingleObject failed:{result}')
        summary_path=HERE/'fixed_bn_4ep/summary.json'
        if not summary_path.exists():
            raise RuntimeError('Existing process ended or was missing without four-epoch summary; no automatic restart')
        summary=json.loads(summary_path.read_text())
        history=json.loads((summary_path.parent/'history.json').read_text())
        assert summary['status']=='completed' and summary['epochs']==4 and summary['optimizer_updates']==876
        assert [r['epoch'] for r in history]==list(range(5))
        assert all(r['bn_buffers_unchanged'] for r in history[1:])
        review=dict(status='program_reviewed',assistant_review='pending',updated=stamp(),
            baseline_passes=summary['baseline']['full_field_passes'],
            selected_passes=summary['selected_validation']['full_field_passes'],
            last_passes=summary['last_validation']['full_field_passes'],
            epochs=[dict(epoch=r['epoch'],passes=r['validation']['full_field_passes'],loss=r.get('loss')) for r in history],
            bn_buffers_unchanged=True,optimizer_updates=876)
        atomic_json(HERE/'fixed_bn_program_review.json',review)
        if review['selected_passes']>review['baseline_passes']:
            state('awaiting_assistant_review',reason='Fixed-BN candidate improved full-validation recovery; review before choosing continuation')
            return
        acceptance=HERE/'trajectory_branch_acceptance.json'
        if not acceptance.exists():
            state('starting_real_trajectory_check',reason='Fixed BN did not improve recovery in four epochs')
            if not run_child('trajectory_check',['check_trajectory_nikon.py']):return
            raise RuntimeError('Inspect the real derivative result and register branch acceptance before training')
        check=json.loads(acceptance.read_text())
        assert check['status']=='passed'
        if not run_child('trajectory_train',['train_four.py','--trial','trajectory_4ep']):return
        if (HERE/'STOP').exists():state('paused',stage='trajectory_train');return
        result=json.loads((HERE/'trajectory_4ep/summary.json').read_text())
        assert result['status']=='completed' and result['epochs']==4
        state('completed',assistant_review='pending',trial='trajectory_4ep',
            selected_passes=result['selected_validation']['full_field_passes'],last_passes=result['last_validation']['full_field_passes'])
    except Exception:
        state('failed',error=traceback.format_exc());raise
    finally:
        if handle:kernel.CloseHandle(handle)
        lock.close()


if __name__=='__main__':main()
