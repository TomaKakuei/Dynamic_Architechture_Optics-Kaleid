"""Attach current J once; serial bounded follow-up, no duplicate GPU jobs."""
import argparse,ctypes,json,os,subprocess,sys,traceback,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
KERNEL=ctypes.WinDLL('kernel32',use_last_error=True)
KERNEL.OpenProcess.argtypes=[ctypes.c_uint32,ctypes.c_int,ctypes.c_uint32]
KERNEL.OpenProcess.restype=ctypes.c_void_p
KERNEL.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32]
KERNEL.CloseHandle.argtypes=[ctypes.c_void_p]


def write(path,value):
    temp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');os.replace(temp,path)


def state(stage,**kw):
    write(ROOT/'coordinator_state.json',dict(stage=stage,pid=os.getpid(),updated=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))


def stopped():return (ROOT/'STOP_REQUESTED.json').exists()


def result(tag):
    path=ROOT/tag/'result.json'
    if not path.exists():raise RuntimeError('Incomplete stage: '+tag)
    return json.loads(path.read_text())


def gate(r):
    return bool((r['gained_over_loop'] or r['center_gained']) and not(r['lost_loop'] or r['center_lost'] or r['new_invalid']))


def run(tag,rank,cycles,split='development',adaptive=False):
    if (ROOT/tag/'result.json').exists():return result(tag)
    if stopped():raise InterruptedError('Stop marker prevents follow-up')
    cmd=[sys.executable,'-X','utf8','-u','run_trial.py','--tag',tag,'--fd','central','--rank',str(rank),
         '--cycles',str(cycles),'--split',split]
    if adaptive:cmd.append('--adaptive')
    with (ROOT/(tag+'.stdout.log')).open('a',encoding='utf-8') as out,(ROOT/(tag+'.stderr.log')).open('a',encoding='utf-8') as err:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        state('running_stage',tag=tag,child_pid=child.pid,command=cmd)
        code=child.wait()
    if code:raise RuntimeError('Stage exited '+str(code)+': '+tag)
    if stopped():raise InterruptedError('Stop marker prevents follow-up')
    return result(tag)


def main(pid):
    KERNEL.SetThreadExecutionState(0x80000001)
    try:
        state('attached_to_existing_j',child_pid=pid,tag='development_j_central8')
        if not (ROOT/'development_j_central8/result.json').exists():
            handle=KERNEL.OpenProcess(0x00100000,False,pid)
            if not handle:raise OSError(ctypes.get_last_error(),'Could not attach existing J')
            try:
                while KERNEL.WaitForSingleObject(handle,2000)==258:
                    if stopped():raise InterruptedError('Stop marker prevents follow-up')
            finally:KERNEL.CloseHandle(handle)
        if stopped():raise InterruptedError('Stop marker prevents follow-up')
        j=result('development_j_central8');chosen=None
        write(ROOT/'j_gate.json',dict(extend=gate(j),result=j))
        if gate(j):chosen=('j_central8',8,3,False)
        else:
            k=run('development_k_adaptive8',8,3,adaptive=True)
            write(ROOT/'k_gate.json',dict(extend=gate(k),result=k))
            if gate(k):chosen=('k_adaptive8',8,3,True)
            else:
                l=run('development_l_central12',12,2)
                write(ROOT/'l_gate.json',dict(extend=gate(l),result=l))
                if gate(l):chosen=('l_central12',12,2,False)
        final=None
        if chosen:
            name,rank,cycles,adaptive=chosen
            write(ROOT/'frozen_candidate.json',dict(variant=name,rank=rank,cycles=cycles,fd='central',
                center_limit=.01,adaptive=adaptive,radius=.2,scope='Exploratory previously observed32, not independent confirmation'))
            final=run('paired32_'+name,rank,cycles,'confirmation',adaptive)
        state('computations_completed',selected=chosen,paired32=final,assistant_interpretation_pending=True,
              automatic_followup=False,automatic_deployment=False)
    finally:KERNEL.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attach',type=int,required=True)
    try:main(p.parse_args().attach)
    except InterruptedError as error:state('stopped',reason=str(error))
    except BaseException:state('failed',error=traceback.format_exc());raise
