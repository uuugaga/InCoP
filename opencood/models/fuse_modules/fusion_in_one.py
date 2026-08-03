# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
"""Compatibility exports for cooperative BEV fusion modules.

The implementations live in focused modules in this package. Keep this file as
stable import surface for existing configs and models that import from
opencood.models.fuse_modules.fusion_in_one.
"""

from opencood.models.fuse_modules.common import (
    regroup,
    warp_feature,
)

from opencood.models.fuse_modules.ours import ComplementarityGuidedCLCFusion

from opencood.models.fuse_modules.ermvp import ERMVPFusion
from opencood.models.fuse_modules.cobevt import CoBEVT
from opencood.models.fuse_modules.v2xvit import V2XViTFusion
from opencood.models.fuse_modules.where2comm import Where2commFusion

__all__ = [
    "regroup",
    "warp_feature",
    "ComplementarityGuidedCLCFusion",
    "ERMVPFusion",
    "V2XViTFusion",
    "CoBEVT",
    "Where2commFusion",
]
