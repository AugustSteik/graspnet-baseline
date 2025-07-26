""" PLOT EVERYTHING NICELY
    This script is used to plot the point cloud, grasps, objects and any explanations in the scene in a nice way.
"""
import os
from graspnetAPI import GraspNet, Grasp, GraspGroup
import open3d as o3d
import cv2
import sys
import torch
import numpy as np
import json


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from record_something import log_variable, varname
# from custom_test import inference, detatch_dict
from explain_grasps import get_fbi, tuple_to_pointcloud
from inference_for_lime import inference
from xai_test_bed.analysis import generate_explanations, generate_pc_data, calculate_metrics
from xai_test_bed.grad_cam import generate_grad_cam_explanations

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
GNET_ROOT = os.path.dirname(DATASET_ROOT)

extracted = {}

def normalise(arr):
    """
    Normalise the input array to the range [0, 1].
    
    Args:
        arr (np.ndarray): Input array to normalize.
    
    Returns:
        np.ndarray: Normalized array.
    """
    min_val = np.min(arr)
    max_val = np.max(arr)
    
    if max_val - min_val == 0:
        return np.zeros_like(arr)
    
    return (arr - min_val) / (max_val - min_val)

def extract_seed_points_and_scores(module, input, output):
    """ Hook function to extract the seed points and objectness score from the model.
    """
    extracted.update({'net.view_estimator.backbone': output})  # Pointnet extracted features are also stored in end_opints under fp2_features

def check_objectness_scores(objectness_scores, threshold):
    """ Check the objectness probabilities to see if they make sense. """
    probabilities = torch.sigmoid(objectness_scores.squeeze())
    grasp = probabilities[0, :][torch.where(probabilities[0, :] > threshold)]
    no_grasp = probabilities[1, :][torch.where(probabilities[1, :] >= 1- threshold)]
    objectness_pred = torch.argmax(objectness_scores, 1)
    print(len(grasp), len(no_grasp))
    return

def generate_grasp_views(pc, colors, N=10, phi=(np.sqrt(5)-1)/2, r=0.002):
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

def enlarge_points(pc, N=50, phi=(np.sqrt(5)-1)/2, r=0.005):
    """ Perturb points around each point in the point cloud.
    Args:
        pc (np.ndarray): The input point cloud.
        colors (np.ndarray): The colors for each point.
        N (int): Number of views to generate.
        phi (float): Golden ratio.
        center (np.ndarray): Center point for the views.
        r (float): Radius for the views.
    Returns:
        pc
    """
    all_points = []
    all_point_colours = []
    for point, color in zip(np.array(pc.points), np.array(pc.colors)):
        for i in range(N):
            zi = (2 * i + 1) / N - 1
            xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
            yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
            all_points.append(r * np.array([xi, yi, zi]) + point)
            all_point_colours.append(color)
    pc.points = o3d.utility.Vector3dVector(np.array(all_points))
    pc.colors = o3d.utility.Vector3dVector(np.array(all_point_colours))
    return pc

def filter_pointcloud_by_color(point_cloud, color_condition):
    """
    Remove points from a point cloud based on their color.

    Args:
        point_cloud (o3d.geometry.PointCloud): The input point cloud.
        color_condition (function): A function that takes a color (R, G, B) and returns True if the point should be removed.

    Returns:
        o3d.geometry.PointCloud: The filtered point cloud.
    """
    # Convert points and colors to NumPy arrays
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # Apply the color condition to filter points
    mask = np.array([not color_condition(color) for color in colors])  # Keep points where condition is False

    # Filter points and colors
    filtered_points = points[mask]
    filtered_colors = colors[mask]

    # Create a new point cloud with the filtered points and colors
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_point_cloud


if __name__ == '__main__':
    gnet = GraspNet(DATASET_ROOT, camera='kinect', split='test')
    geometries = []
    
    """ Select the scene and annotation id."""
    annotation_id = 0
    scene_id = 100
    object_id = 48  # For scene 100: 9(scissors),11(strawb),20(screwdriver),29(toy plane part 1),30(toy plane part 2),41(paste),48(camel),52(rhino),58(box),62(pantene)
    lime_obj_id = 48#52 - what was used for line
    
    """ Load scene and object meshes """
    # geometries.append(gnet.loadScenePointCloud(sceneId=scene_id, annId=annotation_id, camera='kinect'))
    geometries += [gnet.loadSceneModel(sceneId=scene_id, annId=annotation_id)[-4]]
    # geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(0.02, (0.05, -0.05, -0.05)))  # triad
    """ Load lime explanations"""
    if False:
        lime_explanations = o3d.io.read_point_cloud(os.path.join(GNET_ROOT, 'grasp_lime/visu', f'scene_0{scene_id}-{lime_obj_id}.ply'))
        if True:  # remove points tha tare not importants
            def color_condition(color):
                return color[0] < 0.2 and color[2] < 0.2
            lime_explanations = filter_pointcloud_by_color(lime_explanations, color_condition)
        lime_explanations = enlarge_points(lime_explanations)
        geometries.append(lime_explanations)
        
    """ Run Eval """
    if True:
        net, gg, end_points, grasps_per_object, best_grasp_per_object, logits, prediction, point_cloud = inference(object_id=lime_obj_id)
        # grasp_array = np.array([grasp.numpy() for grasp in best_grasp_per_object.values()])
        grasp_array = np.array([best_grasp_per_object[lime_obj_id].numpy()])
    
    """ Calculate pointnet feature-based metrics """
    if False:
        fbi_explanations = tuple_to_pointcloud(get_fbi(end_points, type='sum', rm_outliers=True, filter_scene=True))
        geometries.append(fbi_explanations)
    
    """ Correlation explanations """
    if False:
        corrcoef_keys = ['objectness_score_corrcoef', 
                        'top_grasp_score_corrcoef', 
                        'base_case_top_view_ind_grasp_score_corrcoef', 
                        'top_view_ind_corrcoef', 
                        'original_view_grasp_score_change_corrcoef']
        out_dict = calculate_metrics()
        corr_explanations = generate_explanations(end_points['fp2_features'].cpu().squeeze().numpy(),
                                                  out_dict, key='objectness_score_corrcoef',
                                                  global_calc=True, mask_features=True)
        corr_explanations_pc = generate_pc_data(end_points['fp2_xyz'].cpu().squeeze().numpy(),
                                                corr_explanations,
                                                valid_points=out_dict['params']['valid_points'],
                                                asym_norm=True)
        
        if False:  # remove points tha tare not importants
            def color_condition(color):
                return color[0] < 0.2 and color[2] < 0.2
            corr_explanations_pc = filter_pointcloud_by_color(corr_explanations_pc, color_condition)
        corr_explanations_pc = enlarge_points(corr_explanations_pc)
        geometries.append(corr_explanations_pc)
    
    """ Gradient explanations """
    if False:
        objectness_scores, view_scores = generate_grad_cam_explanations()
        grad_explanations_pc = generate_pc_data(end_points['fp2_xyz'].cpu().squeeze().numpy(), normalise(view_scores))
        grad_explanations_pc = enlarge_points(grad_explanations_pc)
        geometries.append(grad_explanations_pc)
    
    """ Add point cloud to plot """
    """ Add selected grasps to plot """
    one_object_grasps = np.array([grasp.numpy() for grasp in grasps_per_object[object_id]])
    best_grasp_group = GraspGroup(grasp_array).to_open3d_geometry_list()
    one_object_group = GraspGroup(one_object_grasps).to_open3d_geometry_list()
    
    # geometries += one_object_group
    geometries += best_grasp_group

    view_ctl_params = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2025-04-21-11-49-43.json")

    vis = o3d.visualization.Visualizer()  # type: ignore
    vis.create_window()
    for g in geometries:
        vis.add_geometry(g)
    view_ctl = vis.get_view_control()
    view_ctl.convert_from_pinhole_camera_parameters(view_ctl_params)
    
    vis.run()
    