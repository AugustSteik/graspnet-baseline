from graspnetAPI import GraspNet, Grasp, GraspGroup
import open3d as o3d
import cv2
import numpy as np
from pathlib import Path
import sys


gnet_root = Path(__file__).parent.parent
dataset_root = gnet_root / 'dataset'
sys.path.append(str(gnet_root))
from dataset.graspnet_dataset import GraspNetDataset

TEST_DATASET = GraspNetDataset(dataset_root, valid_obj_idxs=None, grasp_labels=None, split='test', camera='kinect', num_points=20000, remove_outlier=True, augment=False, load_label=False, scene_list=[0], ann_ids=[0])

print(TEST_DATASET.labelpath)