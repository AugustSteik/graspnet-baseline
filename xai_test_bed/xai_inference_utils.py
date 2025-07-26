import torch
import json
import numpy as np
import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from models.graspnet import GraspNet, pred_decode, pred_decode_per_object, pred_decode_per_object_view_logits
from dataset.graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
from utils.collision_detector import ModelFreeCollisionDetector
from utils.label_generation import process_grasp_labels

ROOT = os.path.dirname(os.path.abspath(__file__))

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


def load_data(data_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(data_path, 'r') as f:
        end_points = json.load(f)

    for key in end_points.keys():
        if end_points[key] is None:
            continue
        if 'list' in key:
            for i in range(len(end_points[key])):
                for j in range(len(end_points[key][i])):
                    end_points[key][i][j] = torch.tensor(end_points[key][i][j]).to(device)
        else:
            end_points[key] = torch.tensor(end_points[key]).to(device)
    return end_points

def load_model(checkpoint_path=cfgs.checkpoint_path):
    """ Load the usual gnet model but return only approachnet, also set it to eval mode.

    Args:
        checkpoint_path (_type_, optional): _description_. Defaults to cfgs.checkpoint_path.

    Returns:
        _type_: _description_
    """
    gnet = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gnet.to(device)
    
    checkpoint = torch.load(checkpoint_path)
    gnet.load_state_dict(checkpoint['model_state_dict'])
    gnet.eval()
    approachnet = gnet.view_estimator.vpmodule
    return approachnet
    
def inference_one_batch(net, end_points, **kwargs):
    """ Runs inference for a single batch on a model and returns the output.
        Can run inference on approachnet or graspnet.

    Args:
        net (_type_): torch model
        end_points (dict): dict containing end points as outputted by the backbone, not dataset
        hook_fn (_type_, optional): _description_. Defaults to None.
    """
    hook, hook_fn = None, None
    if 'hook_fn' in kwargs:
        hook_fn = kwargs['hook_fn']
        hook = net.register_forward_hook(hook_fn)
    with torch.no_grad():  # We may want to keep gradients
        try:
            out = net(end_points)
        except Exception:
            # print(f'Running inference on approachnet with seed_xyz and seed_features from end_points.')
            out = net(end_points['fp2_xyz'], end_points['fp2_features'], end_points)  # approachnet takes: seed_xyz, seed_features, end_points for fwd pass
    if hook_fn is not None:
        hook.remove()  # type: ignore
    out = process_grasp_labels(out) if 'process_grasp_labels' in kwargs else out
    return out

def check_uniqueness(one, two):
    """ Check if end points has changed.

    Args:
        one (_type_): dict before forward pass
        two (_type_): dict after forward pass
    """
    for key in two.keys():
        try:
            _ = one[key]
        except KeyError:
            print(f'{key} not in one.')
            continue
        if isinstance(one[key], torch.Tensor) and isinstance(two[key], torch.Tensor):
            if not torch.equal(one[key], two[key]):
                print(f'Key {key} has changed.')
                
                
def add_labels(end_points):
    annotation_id = 0
    scene_id = 100
    image_range=(annotation_id, annotation_id+1)

    valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root, limited=False)  # SET LOAD_LABEL TO TRUE
    load_label = True
    # valid_obj_idxs, grasp_labels, load_label = None, None, False
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=valid_obj_idxs, grasp_labels=grasp_labels, 
                                split='test', camera=cfgs.camera, num_points=cfgs.num_point, 
                                remove_outlier=True, augment=False, load_label=load_label, debug=True, image_range=image_range, scene_id=scene_id,
                                keep_object_ids=True)
    end_points.update()
    return end_points