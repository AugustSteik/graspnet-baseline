#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 16:27:59 2021

@author: 
"""

import argparse
import numpy as np
import os
import torch
import logging
import sys
# from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import open3d as o3d
import time
from typing import Dict

# import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval, GraspNet as gnet

sys.path.append('/home/as_admin/development/graspnet-baseline')

from models.graspnet import MyGraspNet, pred_decode, GraspNet
from dataset.graspnet_dataset import GraspNetDataset, load_grasp_labels, collate_fn
from utils.collision_detector import ModelFreeCollisionDetector

from grasp_lime.lime import lime_3d_remove

from record_something import log_variable, varname

log_vars = False

LIME_DIR = os.path.dirname(os.path.abspath(__file__))  # LIME directory
GRASPNET_DIR = os.path.dirname(LIME_DIR)  # Graspnet directory

parser = argparse.ArgumentParser('Model')
parser.add_argument('--gpu', type=str, default='cuda', help='specify gpu device')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')  # Seed points
parser.add_argument('--log_dir', type=str, default='logs/', help='Experiment root')
parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
parser.add_argument('--num_votes', type=int, default=1, help='Aggregate classification scores with voting [default: 3]')
parser.add_argument('--model_name', type=str, default='pointnet2_cls_ssg')
parser.add_argument('--normals', type=bool, default=False)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--input_file_path', type=str, default='/home/as_admin/development/graspnet-baseline/LIME-3D/data/ModelNet40/airplane/train/airplane_0001.ply')
parser.add_argument('--dataset_root', required=False, default=f'{GRASPNET_DIR}/dataset', help='Dataset root')
parser.add_argument('--checkpoint_path', required=False, default=f'{GRASPNET_DIR}/logs/original/checkpoint-kn.tar', help='Model checkpoint path')
parser.add_argument('--dump_dir', required=False, default=f'{GRASPNET_DIR}/logs/original/test_outputs', help='Dump dir to save outputs')
parser.add_argument('--camera', required=False, default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_pc_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()

extracted = {}


def take_second(elem):
    return elem[1]

def take_first(elem):
    return elem[0]

def gen_pc_data(ori_data,segments,explain,label,filename):
    """
    NOTE:
    Args:
        ori_data: n (usually 1024) sampled piontcloud points
        segments: int array of length 1024
        explain: dict{key: [tuple, ], } each tuple is a pair of (int, float) and each value in dict is a 20 long list of these
        l: int corresponding to the id of the label?
        filename...
    Returns:
        colourful pc
    """
    basic_path = os.path.join(LIME_DIR, 'visu')
    color = np.zeros([ori_data.shape[0],3])
    max_contri = 0
    min_contri = 0
    for k in explain[label]:
        if k[1] > 0 and k[1] > max_contri:
            max_contri = k[1]
        elif k[1] < 0 and k[1] < min_contri:
            min_contri = k[1]
    if max_contri > 0:
        positive_color_scale = 1/max_contri
    
    else:
        positive_color_scale = 0
    if min_contri < 0:
        negative_color_scale = 1/min_contri
    else:
        negative_color_scale = 0
    ex_sorted = sorted(explain[label],key=take_first,reverse=False)
    for i in range(segments.shape[0]):
        if ex_sorted[segments[i]][1] > 0:
            color[i][0] = ex_sorted[segments[i]][1] * positive_color_scale
        elif ex_sorted[segments[i]][1] < 0:
            color[i][2] = ex_sorted[segments[i]][1] * negative_color_scale
        else:
            color[i] = [0,0,0]
    pc_colored = np.concatenate((ori_data,color),axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_colored[:,0:3])
    pc.colors = o3d.utility.Vector3dVector(pc_colored[:,3:6])
    o3d.io.write_point_cloud(os.path.join(basic_path, filename.rsplit('/')[-1]), pc)
    return
    
def reverse_points(points,segments,explain,start='positive',percentage=0.2):
    num_input_dims = points.shape[1]
    basic_path = os.path.join(LIME_DIR, 'output')
    filename = 'reversed.ply'
    if start == 'positive':
        to_rev_list = np.argsort(explain)[-int(len(explain)*percentage):]
        to_rev_list = to_rev_list[::-1]
    elif start == 'negative':
        to_rev_list = np.argsort(explain)[:int(len(explain)*percentage)]
    else:
        print('Wrong start input!')
        return points
    for i in range(len(to_rev_list)):
        segment_to_rev = to_rev_list[i]
        rev_points_index = np.argwhere(segments==segment_to_rev)
        for p in rev_points_index:
            points[p] = np.zeros([num_input_dims])
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:,0:3])
    o3d.io.write_point_cloud(os.path.join(basic_path, filename), pc)
    return points

def sampling(points, sample_size):
    num_p = points.shape[0]
    index = range(num_p)
    np.random.seed(1)
    sampled_index = np.random.choice(index,size=sample_size)
    sampled = points[sampled_index]
    return sampled

def init_model(hook_fn=None, wrapped=True, scene_id=0, image_range=(0, 1), get_labels=False):
    """Initialses the GraspNet model. 

    Args:
        hook_fn (_type_, optional): Defaults to None.
        wrapped (bool, optional): To wrap or not to wrap? Defaults to True.
        scene_id (int, optional): Defaults to 1.
        image_range (tuple, optional): _description_. Defaults to (0, 1).

    Returns:
        net (GraspNet): GraspNet model instance, can be wrapped or unwrapped.
    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        pass
    
    # Load the dataset and dataloader
    valid_obj_idxs, grasp_labels = None, None
    if get_labels:
        valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
    TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=valid_obj_idxs, grasp_labels=grasp_labels, 
                                split='test', camera=cfgs.camera, num_points=cfgs.num_point, 
                                remove_outlier=True, augment=False, load_label=False, debug=True, image_range=image_range, scene_id=scene_id)
    
    print(len(TEST_DATASET))
    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
        num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
    print(len(TEST_DATALOADER))
    
    # Load model, checkpoint and move to device
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                    cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.to(device)
    net.eval()
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    net = GraspNetWrapper(net) if wrapped else net
    return net, TEST_DATALOADER

class GraspNetWrapper(torch.nn.Module):
    def __init__(self, model):
        super(GraspNetWrapper, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        if isinstance(args[0], dict):
            output_dict = self.model(*args, **kwargs)  # Get the model's raw output
        else:
            output_dict = self.model({'point_clouds': args[0]}, **kwargs)
        grasp_logits = output_dict['objectness_score']
        grasp_probs = torch.sigmoid(grasp_logits)
        top_grasp_probs, top_grasp_idxs = grasp_probs.topk(1, dim=-1)  # Get the top grasp and no-grasp probabilities and indices
        return grasp_logits[:, 0, :], top_grasp_idxs[:, 0]
    
def run_eval(net, dataloader, hook_fn=None, raw_predictions=True):
    """Runs the evaluation on the GraspNet model with an optional hook function.
    If not using raw predictions, a grasp group is returned for the chosen scene.
    Otherwise, logits for graspability and end_points are returned.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hook_handle, logits, predictions = None, None, None
    grasp_preds = []
    if hook_fn:
        try:
            hook_module = net.model.view_estimator.vpmodule
        except Exception:
            hook_module = net.view_estimator.vpmodule
        hook_handle = hook_module.register_forward_hook(hook_fn)
    
    tic = time.time()
    for _, batch_data in enumerate(dataloader):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
                
        with torch.no_grad():
            if raw_predictions:
                logits, end_points = net(batch_data)
                predictions = end_points
            else:
                end_points = net(batch_data)
                grasp_preds = pred_decode_loose(end_points)  # Original does not output any grasps, let alone one for each point
                if log_vars:
                    log_variable(f'inference{_}-custom', varname(grasp_preds), grasp_preds)
                    log_variable(f'inference{_}-custom', varname(end_points), end_points)
                    
    if not raw_predictions:
        for i in range(cfgs.batch_size):
            predictions = GraspGroup(grasp_preds[i].detach().cpu().numpy())  # NOTE: Always working with one batch but kept this from original for clarity.
            
    toc = time.time()
    print('Eval time: %fs'%(toc-tic))
    
    if hook_handle is not None:
        hook_handle.remove()
        
    return logits, predictions

def show_pc(dir):
    """
    Plots coloured PC explanations using matplotlib. 
    RED - more positive contribution to classification
    BLUE - more negative contribution
    DIM POINTS - 0 contribution
    """
    filename = dir.rsplit('/')[-1]
    basic_path = os.path.join(LIME_DIR, 'visu', filename)
    # basic_path = dir
    pc = o3d.io.read_point_cloud(basic_path)

    # Convert to NumPy arrays
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)  # Colors are normalized between 0 and 1

    # Create Matplotlib 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Scatter plot with colors
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=3)
    plt.show()
    return

import plotly.express as px

def show_pc_plx(dir, seed_points, key_points):
    """
    Plots coloured PC explanations using plotly express. 
    RED - more positive contribution to classification
    BLUE - more negative contribution
    DIM POINTS - 0 contribution
    """
    filename = dir.rsplit('/')[-1]
    basic_path = os.path.join(LIME_DIR, 'visu', filename)
    
    pc = o3d.io.read_point_cloud(basic_path)

    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)  # Colors are normalised between 0 and 1

    # Create a DataFrame for Plotly
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df['color_r'] = colors[:, 0]
    df['color_g'] = colors[:, 1]
    df['color_b'] = colors[:, 2]
    df['color'] = df.apply(lambda row: f'rgb({row.color_r*255}, {row.color_g*255}, {row.color_b*255})', axis=1)
    
    for idx, key_point in enumerate(key_points):
        key_point_xyz = seed_points[key_point]
        new_point_df = pd.DataFrame([key_point_xyz], columns=['x', 'y', 'z'])
        new_point_df['color_r'] = 1.0 - idx/10
        new_point_df['color_g'] = 0.063
        new_point_df['color_b'] = 0.941
        new_point_df['color'] = new_point_df.apply(lambda row: f'rgb({row.color_r*255}, {row.color_g*255}, {row.color_b*255})', axis=1)
        new_point_df['hover_name'] = "Key Point"
        df = pd.concat([df, new_point_df], ignore_index=True)
        
    # Plot using Plotly Express
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', color_discrete_map="identity")
    fig.show()


def detatch_dict(in_dict):
    """ Detatch tensors in dict and load them on host memory.
    """
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
            out_dict[k] = v.detach().cpu()
        else:
            out_dict[k] = v
    return out_dict

def extract_seed_points_and_scores(module, input, output):
    """Hook function to extract the seed points and objectness score from the model.
    """
    extracted.update(detatch_dict(output))
    # extracted['seed_xyz'] = end_points['input_xyz']  # (1, 1024, 3)
    # extracted['objectness_score'] = end_points['objectness_score']  # (1, 2, 1024)
    # for i in range(extracted['objectness_score'].shape[2]):
    #     m = nn.Sigmoid()
    #     preds = m(extracted['objectness_score'][0, :, i].detach())
    #     if preds[0] > preds[1]:
    #         print('Grasp! %f, %f' % (preds[0], preds[1]))
    #     else:
    #         print('%f, %f' % (preds[0], preds[1]))

if __name__ == '__main__':
    # Set up logging
    def log_string(str):
        logger.info(str)
        print(str)

    logger = logging.getLogger("GraspNet_Explanation")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % os.path.join(LIME_DIR, cfgs.log_dir))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(cfgs)
    # Model initialisation
    scene_id = 100
    net, dataloader = init_model(hook_fn=extract_seed_points_and_scores, wrapped=True, scene_id=scene_id, image_range=(0, 1))
    logits, predictions = run_eval(net, dataloader, hook_fn=extract_seed_points_and_scores)
    
    points = np.asarray(extracted.get('input_xyz', {}).squeeze())  # (1024, 3), extracted seed points used for explanations
    
    explainer = lime_3d_remove.LimeImageExplainer(random_state=0)
    
    tmp = time.time()
    explanation = explainer.explain_instance(points, net, top_labels=5, num_features=50, num_samples=20, random_seed=0)
    print ('Completed in: ',time.time() - tmp,'s')
    gen_pc_data(points, explanation.segments, explanation.local_exp, predictions.item(), f'/scene_{str(scene_id).zfill(4)}.ply')

    show_pc_plx(f'/scene_{str(scene_id).zfill(4)}.ply', points, explanation.top_labels) # type: ignore
    
    g = gnet('/home/as_admin/development/graspnet-baseline/dataset', camera='kinect', split='test')
    # g.show6DPose(sceneIds = scene_id, show = True)
    g.showSceneGrasp(sceneId = scene_id, camera = 'kinect', annId = 0, format = '6d')
    