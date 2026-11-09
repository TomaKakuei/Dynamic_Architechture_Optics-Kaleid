from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
SOURCE=WORKSPACE/'RAYOPTICS_RESTORE_20260929/dao_mod03/sampling_front3_rear4/olympus_joint_20261015'
sys.path.insert(0,str(SOURCE))
import common
from data import RelayCamera,Blocks
from tensor_optics import TensorOptics
from network import restore
from objective import Objective
import numpy as np
import torch
sys.path.insert(0,str(HERE))


def state(stage,**kw):common.atomic(HERE/'state.json',dict(stage=stage,updated=common.stamp(),**kw))


def expected_adu(image):return 108.+16000.*image/image.max().clamp_min(1e-30)


def sigma_from_adu(image):
    signal=(image-108.).clamp_min(0)
    return (4*signal+16*(1.2**2+.5**2)+(.005*signal)**2+1/12).sqrt()
