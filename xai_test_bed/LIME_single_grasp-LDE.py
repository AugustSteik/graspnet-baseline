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

# import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval, GraspNet as gnet

GRASPNET_DIR = os.path.dirname(os.path.dirname(__file__))
LIME_DIR = os.path.join(GRASPNET_DIR, 'grasp_lime')
ROOT = os.path.dirname(__file__)

sys.path.append(ROOT)
sys.path.append(GRASPNET_DIR)

# from models.graspnet import MyGraspNet, pred_decode, GraspNet
# from dataset.graspnet_dataset import GraspNetDataset, load_grasp_labels, collate_fn
# from utils.collision_detector import ModelFreeCollisionDetector

from grasp_lime.lime import lime_3d_remove

from record_something import log_variable, varname
from inference_for_lime import inference
from xai_inference_utils import load_data, load_model, inference_one_batch


log_vars = False

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

import os
import numpy as np
import open3d as o3d

def gen_pc_dater(ori_data, segments, explain, label, filename, thresholds=(95, 5)):
    """
    Generate a colored point cloud visualization with nonlinear color scaling.

    Args:
        ori_data (numpy.ndarray): n (usually 1024) sampled point cloud points.
        segments (numpy.ndarray): Integer array of length 1024 mapping each point to a segment.
        explain (dict): Dictionary {key: [(int, float), ...]} where each key corresponds to 
                        a label and each value is a list of (segment_id, contribution).
        label (int): Corresponding ID of the label.
        filename (str): Output filename for the visualization.

    Returns:
        None. Saves the colored point cloud visualization.
    """
    basic_path = os.path.join(LIME_DIR, 'visu')

    color = np.zeros((ori_data.shape[0], 3))

    # Extract contribution values
    contributions = np.array([k[1] for k in explain[label]])

    # Compute percentiles for non-linear scaling
    pos_threshold = np.percentile(contributions[contributions > 0], thresholds[0]) if np.any(contributions > 0) else 0
    neg_threshold = np.percentile(contributions[contributions < 0], thresholds[1]) if np.any(contributions < 0) else 0

    # Normalize scaling factors
    max_contri = np.max(contributions) if np.any(contributions > 0) else 1
    min_contri = np.min(contributions) if np.any(contributions < 0) else -1

    def nonlinear_scale(value, threshold, max_value):
        """ 
        Apply a nonlinear scaling to emphasize high contributions.
        If a value is above the threshold, it gets full intensity.
        Otherwise, it is scaled quadratically for smooth dimming.
        """
        if abs(value) >= abs(threshold):
            return 1.0  # Highlight top contributions
        else:
            return (value / max_value) ** 2  # Quadratic scaling for dimming

    # Sort explanations by segment ID
    ex_sorted = sorted(explain[label], key=lambda x: x[0])

    # Assign colors to points based on their segment contributions
    for i in range(segments.shape[0]):
        seg_index = segments[i]
        contri_value = ex_sorted[seg_index][1]

        if contri_value > 0:
            color[i][0] = nonlinear_scale(contri_value, pos_threshold, max_contri)  # Red for positive
        elif contri_value < 0:
            color[i][2] = nonlinear_scale(contri_value, neg_threshold, min_contri)  # Blue for negative
        else:
            color[i] = [0, 0, 0]  # Neutral

    # Create a colored point cloud
    pc_colored = np.concatenate((ori_data, color), axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pc_colored[:, :3])
    pc.colors = o3d.utility.Vector3dVector(pc_colored[:, 3:6])

    # Save the point cloud
    output_path = os.path.join(basic_path, os.path.basename(filename))
    o3d.io.write_point_cloud(output_path, pc)

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

def show_pc_plx(dir, seed_points, key_points=None):
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
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])  # NOTE
    df['color_r'] = colors[:, 0]
    df['color_g'] = colors[:, 1]
    df['color_b'] = colors[:, 2]
    df['color'] = df.apply(lambda row: f'rgb({row.color_r*255}, {row.color_g*255}, {row.color_b*255})', axis=1)
    
    if key_points is not None:
        for idx, key_point in enumerate(key_points):
            # key_point_xyz = seed_points[key_point]
            new_point_df = pd.DataFrame([key_point], columns=['x', 'y', 'z'])
            new_point_df['color_r'] = 0.5
            new_point_df['color_g'] = 1.0 - idx/len(key_points)
            new_point_df['color_b'] = 0.5
            new_point_df['color'] = new_point_df.apply(lambda row: f'rgb({row.color_r*255}, {row.color_g*255}, {row.color_b*255})', axis=1)
            new_point_df['hover_name'] = "Key Point"
            df = pd.concat([df, new_point_df], ignore_index=True)
        
    # Plot using Plotly Express
    fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', color_discrete_map="identity")
    fig.show()

# def find_n_closest_points(anchor, points, n):  INSTEAD, maybe just order all the points in the dict to begin with!?
#     """ Find n closest points to anchor in 3D space. 

#     Args:
#         anchor (tensor): coorrdinates
#         points (list(tensor)): points to select these from 
#         n (int): number of points
#     """

def reorder_points_nearest_neighbor_torch(pc):  # Kind of like a random walk sim :)
    """
    Reorder points so that consecutive points in the returned tensor
    are physically close in Euclidean space (PyTorch version).

    Arguments:
        pc: torch.Tensor of shape (N, 3).

    Returns:
        ordered_pc: torch.Tensor of shape (N, 3) with points reordered.
        idx_order:  List[int] with the permutation of indices.
    """
    pc = pc.cpu()  # Make sure on CPU for easy indexing
    N = pc.shape[0]

    visited = set([0])
    idx_order = [0]

    for _ in range(N - 1):
        current_idx = idx_order[-1]
        current_point = pc[current_idx]
        # (N, 3) - (1, 3) -> (N, 3), then sum of squared differences
        dists = torch.sum((pc - current_point) ** 2, dim=1)

        # Mark visited points as "infinite distance"
        dists[list(visited)] = float('inf')

        next_idx = torch.argmin(dists)
        idx_order.append(next_idx.item())
        visited.add(next_idx.item())

    ordered_pc = pc[idx_order]
    return ordered_pc, idx_order




if __name__ == '__main__':
    # # Set up logging
    # def log_string(str):
    #     logger.info(str)
    #     print(str)

    # logger = logging.getLogger("GraspNet_Explanation")
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('%s/eval.txt' % os.path.join(LIME_DIR, cfgs.log_dir))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    # log_string('PARAMETER ...')
    # log_string(cfgs)
    
    """NOTE:  model ouputs: logits for each of the 40 classes [1, 40], sa3 layer outputs!? [1, 1024, 1]
              Single tensor[] of predictions is only needed for their genpcdata function"""
    
    scene_id = 100
    object_id = 48  # For scene 100: 9(scissors),11,20,29,30,41(paste),48(camel),52(elephant),58(box),62
    
    net, gg, end_points, grasps_per_object, best_grasp_per_object, logits, prediction, point_cloud = inference(object_id=object_id)
    
    top_grasp_xyz = best_grasp_per_object[object_id][13:16].cpu().numpy()
    
    point_cloud = point_cloud.squeeze(0)
    explainer = lime_3d_remove.LimeImageExplainer(random_state=0)
    
    
    num_points = 5  # doubles 
    point_idx, view_idx = prediction if prediction else (0, 0)
    
    key_points = end_points['fp2_xyz'][:, point_idx, :].squeeze().cpu().numpy()  # They are all over the place even though they are next to eachother in the list
    # key_points, inds = sort_by_xyz(end_points['fp2_xyz'].squeeze())
    flattened_idx = (300 * point_idx) + view_idx
    # flattened_idx = torch.argmax(logits)
    labels = tuple(idx*300+view_idx for idx in range(point_idx-num_points, point_idx+num_points, 1))
    
    # best_score_point_idx = (torch.argmax(logits).item() - view_idx) / 300
    
    tmp = time.time()
    explanation = explainer.explain_instance(point_cloud, net, labels=(flattened_idx, ), num_features=500, num_samples=10, random_seed=0)  # 3k features no work
    print ('Completed in: ',time.time() - tmp,'s')
    
    gen_pc_dater(point_cloud.cpu().numpy(), explanation.segments, explanation.local_exp,flattened_idx, f'/scene_0{scene_id}.ply')
    show_pc_plx(f'/scene_0{scene_id}.ply', point_cloud.cpu().numpy(), [key_points]) # type: ignore   # WAS top_grasp
    