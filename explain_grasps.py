""" Testing for GraspNet baseline model. """

import os
import sys
import numpy as np
import argparse
import time

# import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
# import plotly.offline as pyo

import torch
from torch.utils.data import DataLoader
from graspnetAPI import GraspGroup, GraspNetEval

from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from models.graspnet import MyGraspNet, pred_decode, GraspNet
from dataset.graspnet_dataset import GraspNetDataset, collate_fn
from utils.collision_detector import ModelFreeCollisionDetector

from graspnetAPI import GraspNet as gnet

from test_bed.rotation import rotate_matrix
from pc_dataset import PCDataset

# ------------------------------------------------------------------------- GLOBAL CONFIG BEGIN
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=False, default=f'{ROOT_DIR}/dataset', help='Dataset root')
parser.add_argument('--checkpoint_path', required=False, default=f'{ROOT_DIR}/logs/original/checkpoint-kn.tar', help='Model checkpoint path')
parser.add_argument('--dump_dir', required=False, default=f'{ROOT_DIR}/logs/original/test_outputs', help='Dump dir to save outputs')
parser.add_argument('--camera', required=False, default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers used in evaluation [default: 30]')
cfgs = parser.parse_args()

extracted = {}  # Global var so it can be modified by the hook function

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def initnetandruneval(hook_fn=None, scene_id=1, image_range=(0, 1), object_ids=None):
    """ Initialize the network and run evaluation on the dataset. """
    
    hook_handle = None
    
    if not os.path.exists(cfgs.dump_dir): os.mkdir(cfgs.dump_dir)

    # Init datasets and dataloaders 
    def my_worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        pass

    # Create Dataset and Dataloader
    if object_ids is None:
        split = 'test'  # with debug=True passed to dataset only scene 0 is selected. 
        TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs=None, grasp_labels=None, 
                                    split=split, camera=cfgs.camera, num_points=cfgs.num_point, 
                                    remove_outlier=True, augment=False, load_label=False, debug=True, image_range=image_range, scene_id=scene_id)
    else:
        TEST_DATASET = PCDataset(object_ids=object_ids)
    print(len(TEST_DATASET))
    # SCENE_LIST = TEST_DATASET.scene_list()
    TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=cfgs.batch_size, shuffle=False,
        num_workers=2, worker_init_fn=my_worker_init_fn, collate_fn=collate_fn)
    print(len(TEST_DATALOADER))
    # Init the model
    # net = MyGraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
    #                      cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()
    
    if hook_fn is not None:
        hook_module = net.view_estimator.backbone
        hook_handle = hook_module.register_forward_hook(hook_fn)
        
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])

    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))

    batch_interval = 100
    
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            _ = net(batch_data)  # end_points
        #     grasp_preds = pred_decode(end_points)
        
        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs'%(batch_idx, (toc-tic)/batch_interval))
            tic = time.time()
    
    if hook_handle is not None:
        hook_handle.remove()
        
    return

def get_fbi(seed_xyz, seed_features, type='sum', rm_outliers=True):
    """ Calculates feature based importance (L1 Norm) of seed points from PointNet2 backbone output (before vp module layer).
    """
    if len(seed_features.shape) > 2:
        seed_features = seed_features.squeeze()
    if type.lower().strip() == 'sum':
        explanations = torch.sum(torch.abs(seed_features), dim=0, keepdim=True)
    elif type.lower().strip() == 'var':
        explanations = torch.var(seed_features, dim=0, keepdim=True)
    else:
        print("Invalid type. Using default: var")
        explanations = torch.var(seed_features, dim=0, keepdim=True)
    if rm_outliers:
        explanations = torch.where(explanations > (1.5 * torch.mean(explanations)), torch.mean(explanations), explanations)
    # explanations = _normalised_to_heatmap((explanations - explanations.min()) / (explanations.max() - explanations.min()))
    explanations = torch.zeros(explanations.shape)
    return seed_xyz.squeeze(), explanations.squeeze()

# def _normalised_to_heatmap(values):
#     values = values.squeeze().cpu().numpy() if isinstance(values, torch.Tensor) else values
#     colours = np.array([[value, 0, 0] for value in values])
#     return 
def _normalised_to_heatmap(values):
    """
    Map normalized values to a heatmap color gradient (deep blue to bright yellow).
    
    Args:
        values (torch.Tensor or np.ndarray): Normalized values in the range [0, 1].
    
    Returns:
        np.ndarray: Colors mapped to the heatmap gradient.
    """
    values = values.squeeze().cpu().numpy() if isinstance(values, torch.Tensor) else values

    # Map values to a gradient from deep blue (low values) to bright yellow (high values)
    colours = np.zeros((len(values), 3))  # Initialize RGB array
    colours[:, 0] = values  # Red channel increases with value
    colours[:, 1] = values  # Green channel increases with value
    colours[:, 2] = 1 - values  # Blue channel decreases with value

    return colours

def tuple_to_pointcloud(tensor_tuple):
    """
    Convert a tuple of tensors (coordinates and normalized colors) into an Open3D PointCloud.
    
    Args:
        tensor_tuple (tuple): A tuple containing two tensors:
                              - tensor_tuple[0]: Coordinates (shape: Nx3)
                              - tensor_tuple[1]: Normalized colors (shape: Nx3, values in [0, 1])
    
    Returns:
        o3d.geometry.PointCloud: The resulting Open3D PointCloud.
    """
    # Extract coordinates and colors from the tuple
    coords, colors = tensor_tuple

    # Ensure the tensors are on the CPU and convert them to NumPy arrays
    coords_np = coords.cpu().numpy() if isinstance(coords, torch.Tensor) else coords
    colors_np = colors.cpu().numpy() if isinstance(colors, torch.Tensor) else colors
    
    
    # Create an Open3D PointCloud object
    point_cloud = o3d.geometry.PointCloud()

    # Assign points and colors to the PointCloud
    point_cloud.points, point_cloud.colors = generate_grasp_views(coords_np, colors_np)

    return point_cloud

def generate_grasp_views(pc, colors, N=100, phi=(np.sqrt(5)-1)/2, r=0.005):
    """ Perturb points around each point in the point cloud.
    Args:
        pc (np.ndarray): The input point cloud.
        colors (np.ndarray): The colors for each point.
        N (int): Number of views to generate.
        phi (float): Golden ratio.
        center (np.ndarray): Center point for the views.
        r (float): Radius for the views.
    Returns:
        np.ndarray: Generated grasp views.
        np.ndarray: Colors for the generated grasp views.
    """
    all_points = []
    all_point_colours = []
    for point, color in zip(pc, colors):
        for i in range(N):
            zi = (2 * i + 1) / N - 1
            xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
            yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
            all_points.append(r * np.array([xi, yi, zi]) + point)
            all_point_colours.append(color)
    return o3d.utility.Vector3dVector(np.array(all_points)), o3d.utility.Vector3dVector(np.array(all_point_colours))

def feature_prevelance(seed_features):
    fp = {k: 0 for k in range(256)}
    if seed_features.shape[0] == 1:
        seed_features.squeeze(0)
    for i, point in enumerate(seed_features):  # USE torch...
        for j, feature in point:
            fp[j] += feature
    
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

def show_pc_plotly(pointcloud, explanations=torch.zeros((1024)), seed_xyz=torch.zeros((1024, 3))):
    """
    Plots coloured PC explanations using matplotlib. 
    RED - more positive contribution to classification
    BLUE - more negative contribution
    DIM POINTS - 0 contribution
    """
    explanations_normalised = (explanations - explanations.min()) / (explanations.max() - explanations.min())

    seed_xyz = seed_xyz.numpy()
    pointcloud = pointcloud.numpy()
    
    df_explanations = pd.DataFrame({
        'x': seed_xyz[:, 0],
        'y': seed_xyz[:, 1],
        'z': seed_xyz[:, 2],
        'fbi': explanations_normalised
    })
    
    df_pointcloud = pd.DataFrame({
        'x': pointcloud[:, 0],
        'y': pointcloud[:, 1],
        'z': pointcloud[:, 2]
    })
    
    fig = px.scatter_3d(
        df_explanations,
        x='x',
        y='y',
        z='z',
        color='fbi',
        color_continuous_scale='viridis',
        title="FBI #explanations"
    )
    
    fig.update_traces(marker=dict(size=10))
    
    fig.add_trace(go.Scatter3d(
        x=df_pointcloud['x'],
        y=df_pointcloud['y'],
        z=df_pointcloud['z'],
        mode='markers',
        marker=dict(
            size=1,
            color='black'
        ),
        name="Scene PC"
    ))

    fig.show()
    return

# GPT code:
def show_pc_open3d(pointcloud, explanations=torch.zeros((1024)), seed_xyz=torch.zeros((1024, 3))):
    """
    Plots colored PC explanations using Open3D.
    
    RED - more positive contribution to classification
    BLUE - more negative contribution
    DIM POINTS - 0 contribution
    """
    seed_xyz = seed_xyz.numpy() if isinstance(seed_xyz, torch.Tensor) else seed_xyz
    pointcloud = pointcloud.numpy() if isinstance(pointcloud, torch.Tensor) else pointcloud
    explanations = explanations.numpy() if isinstance(explanations, torch.Tensor) else explanations

    explanations_normalized = (explanations - explanations.min()) / (explanations.max() - explanations.min())
    
    # Map explanations to colors (from blue to red)
    colors = np.zeros((seed_xyz.shape[0], 3))
    colors[:, 0] = explanations_normalized  # Red intensity, greater magnitude
    colors[:, 2] = 1 - explanations_normalized  # Blue intensity lower magnitude

    seed_pc = o3d.geometry.PointCloud()
    seed_pc.points = o3d.utility.Vector3dVector(seed_xyz)
    seed_pc.colors = o3d.utility.Vector3dVector(colors)

    # scene_pc = o3d.geometry.PointCloud()
    # scene_pc.points = o3d.utility.Vector3dVector(pointcloud)
    # scene_pc.paint_uniform_color([0.8, 0.8, 0.8])  # Scene points in grey

    o3d.visualization.draw_geometries([seed_pc], window_name="3D Explanations Visualization")

    return

def extract_hook(module, input, output):
    # Assume output is a tuple: (seed_features, seed_xyz)
    
    seed_features, seed_xyz, end_points = output
    extracted['seed_features'] = seed_features.detach().squeeze()  # Detach to avoid gradient tracking
    extracted['seed_xyz'] = seed_xyz.detach().squeeze()
    extracted['end_points'] = detatch_dict(end_points)
    
if __name__=='__main__':

    scene_id = 110
    object_ids = [16]
    
    initnetandruneval(extract_hook, scene_id=scene_id, image_range=(0, 1), object_ids=object_ids)
    
    seed_features = extracted.get('seed_features', torch.Tensor((256, 1024))).cpu()  # 256 x 1024
    seed_xyz = extracted.get('seed_xyz', torch.Tensor((256, 1024))).cpu()  # 1024 x 3
    pointcloud = extracted.get('end_points', {})['point_clouds'].squeeze()  # 20000 x 3
    
    explanations = get_fbi(seed_features, type='var', rm_outliers=False) # 1024
    
    ## GraspNet:
    
    # g = gnet('/home/as_admin/development/graspnet-baseline/dataset', camera='kinect', split='test')
    # # g.show6DPose(sceneIds = scene_id, show = True)
    # g.showSceneGrasp(sceneId = scene_id, camera = 'kinect', annId = 0, format = '6d')
    
    # # show_pc_plotly(pointcloud, explanations, seed_xyz)
    show_pc_plotly(pointcloud, explanations, seed_xyz)
    