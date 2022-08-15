# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: F401
from .base import BaseHead, BaseHeadCom
from .simple_head import GCNHead, I3DHead, SimpleHead, SlowFastHead, TSNHead
from .simple_head import GCNHeadCom, SimpleHeadCom, SimpleHeadGTCN, ParaGTCNHead, SimpleHeadTCN, ParaTCNHead
from .simple_head import SimpleHeadMultiA, SeqGTCNHeadMultiA

from .cigcn_head import GTCNHeadCoco
