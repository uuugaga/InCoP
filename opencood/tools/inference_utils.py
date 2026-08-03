# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import numpy as np
import torch

from opencood.utils.transformation_utils import get_relative_transformation
from opencood.utils.box_utils import create_bbx, project_box3d

def get_cav_box(batch_data):
    """Return small marker boxes and modality names for all participating CAVs."""
    if 'record_len' in batch_data['ego']:
        lidar_pose = batch_data['ego']['lidar_pose'].cpu().numpy()
        cav_count = int(batch_data['ego']['record_len'].item())
        relative_t = get_relative_transformation(lidar_pose)
        agent_modality_list = batch_data['ego']['agent_modality_list']
    else:
        relative_t = []
        agent_modality_list = []
        for cav_data in batch_data.values():
            relative_t.append(cav_data['transformation_matrix'])
            agent_modality_list.append(cav_data['modality_name'])
        cav_count = len(relative_t)
        relative_t = torch.stack(relative_t, dim=0).cpu().numpy()

    extent = [0.2, 0.2, 0.2]
    ego_box = create_bbx(extent).reshape(1, 8, 3)
    ego_box[..., 2] -= 1.2

    box_list = [ego_box]
    for i in range(1, cav_count):
        box_list.append(project_box3d(ego_box, relative_t[i]))
    cav_box_np = np.concatenate(box_list, axis=0)
    return cav_box_np, agent_modality_list
