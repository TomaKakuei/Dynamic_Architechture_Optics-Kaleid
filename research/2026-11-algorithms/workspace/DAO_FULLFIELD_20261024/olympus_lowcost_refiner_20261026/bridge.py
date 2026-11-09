from pathlib import Path
import sys,json,os
RUN=Path(__file__).resolve().parent
REPAIR=RUN.parent/'olympus_repair_4ep_20261026'
sys.path.insert(0,str(REPAIR))
from local import SOURCE,LEGACY,common,np,torch,Blocks,Objective,restore,sigma_from_adu,DiffMetrics
from repaired_inverse import FullCamera
sys.path.insert(0,str(RUN))


def state(stage,**kw):common.atomic(RUN/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))


def baseline():
    with np.load(REPAIR/'evaluation500/predictions.npz') as z:return {k:z[k].copy() for k in z.files}


def poses():
    with np.load(SOURCE/'training/selected_validation.npz') as z:return {int(i):q.copy() for i,q in zip(z['index'],z['initial'])}
