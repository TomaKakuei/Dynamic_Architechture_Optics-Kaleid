from pathlib import Path
import sys,json,os
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'olympus_lowcost_refiner_20261026'
sys.path.insert(0,str(PREVIOUS))
from bridge import REPAIR,SOURCE,LEGACY,common,np,torch,Blocks,DiffMetrics,restore,sigma_from_adu,baseline,poses
from refiner import Refiner
sys.path.insert(0,str(ROOT))


def state(stage,**kw):common.atomic(ROOT/'state.json',dict(stage=stage,updated=common.stamp(),pid=os.getpid(),**kw))
