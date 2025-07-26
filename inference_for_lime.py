""" Inference of GraspNet baseline model for LIME with outputs and no saved files."""

import os
import sys
import numpy as np
import argparse
import time

import json
from PIL import Image
import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

from models.graspnet import GraspNet, pred_decode, pred_decode_per_object, pred_decode_per_object_view_logits
from dataset.graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
from utils.collision_detector import ModelFreeCollisionDetector

from record_something import varname, log_variable

# from xai_inference_utils import load_data, load_model, inference_one_batch
from get_best_predictions import get_best_grasp_per_object
from model_wrapper import WrappedGraspNet

log_vars = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=False, default=f'{ROOT_DIR}/dataset', help='Dataset root')
parser.add_argument('--checkpoint_path', required=False, default=f'{ROOT_DIR}/logs/original/checkpoint-kn.tar', help='Model checkpoint path')
parser.add_argument('--dump_dir', required=False, default=f'{ROOT_DIR}/new_logs/original/test_outputs', help='Dump dir to save outputs')
parser.add_argument('--camera', required=False, default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.0, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader
# TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, split='test', camera=cfgs.camera, num_points=cfgs.num_point, remove_outlier=True, augment=False, load_label=False)
# get_labels = False


annotation_id = 0
scene_id = 100
image_range=(annotation_id, annotation_id+1)

valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root, limited=True)  # SET LOAD_LABEL TO TRUE
load_label = True
# valid_obj_idxs, grasp_labels, load_label = None, None, False
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=valid_obj_idxs, grasp_labels=grasp_labels, 
                            split='test', camera=cfgs.camera, num_points=cfgs.num_point, 
                            remove_outlier=True, augment=False, load_label=load_label, debug=True, image_range=image_range, scene_id=scene_id,
                            keep_object_ids=True)
print(len(TEST_DATASET))
SCENE_LIST = TEST_DATASET.scene_list()
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
    num_workers=4, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
print(len(TEST_DATALOADER))
# # Init the model
gnet = WrappedGraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                     cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
# gnet = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                    #  cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gnet.to(device)
# Load checkpoint
checkpoint = torch.load(cfgs.checkpoint_path)
gnet.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def inference(net=gnet, dataset=TEST_DATASET, dataloader=TEST_DATALOADER, object_id=9):
    net.eval()
    for batch_idx, batch_data in enumerate(dataloader):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        
        point_cloud=batch_data['point_clouds']
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data, object_id=object_id)
            # end_points = net(batch_data)
            logits, prediction = net(point_cloud)
            # logits, prediction = None, None
            grasp_preds = pred_decode(end_points)
            if log_vars:
                log_variable(f'inference{batch_idx}', varname(grasp_preds), grasp_preds)
                log_variable(f'inference{batch_idx}', varname(end_points), end_points)

        # Dump results for evaluation
        for i in range(cfgs.batch_size):
            data_idx = batch_idx * cfgs.batch_size + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # collision detection
            if cfgs.collision_thresh > 0:
                cloud, _ = dataset.get_data(data_idx, return_raw_cloud=True)
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg = gg[~collision_mask]
                
    grasps_per_object, best_grasp_per_object, view_logits_per_object, best_grasp_view_logits_per_object, point_idxs_per_obj = pred_decode_per_object_view_logits(end_points)
    # best_grasps = get_best_grasp_per_object(per_object_grasps)
    return net, gg, end_points, grasps_per_object, best_grasp_per_object, logits, prediction, point_cloud

if __name__=='__main__':
    net, gg, end_points, grasps_per_object, best_grasp_per_object, logits, prediction, point_cloud = inference()
    print('')
    