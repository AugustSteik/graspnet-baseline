import os
import sys
import torch
import numpy as np

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from models.graspnet import pred_decode_per_object


def get_best_grasp_per_object(grouped_grasps):
    top_grasps = {}
    for object, grasps in grouped_grasps.items():
        if object == -1:
            continue
        scores = torch.tensor([grasp[0] for grasp in grasps])
        top_grasps[object] = grasps[torch.argmax(scores)]
    return top_grasps

