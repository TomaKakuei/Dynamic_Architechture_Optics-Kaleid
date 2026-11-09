from pathlib import Path
import sys,json,os
RUN=Path(__file__).resolve().parent
LEGACY=RUN.parent/'olympus_fullpacket_20261026'
sys.path.insert(0,str(LEGACY))
from support import SOURCE,common,np,torch,Blocks,RelayCamera,TensorOptics,Objective,restore,sigma_from_adu
from camera_inverse import CameraModel
from diff_metrics import DiffMetrics
sys.path.insert(0,str(RUN))


def state(stage,**kw):
    common.atomic(RUN/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))


def save(path,payload):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(payload,tmp);common.commit(tmp,path)
