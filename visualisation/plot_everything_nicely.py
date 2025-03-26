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

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from record_something import log_variable, varname
# from custom_test import inference, detatch_dict
from explain_grasps import get_fbi, tuple_to_pointcloud
from inference_for_lime import inference

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
GNET_ROOT = os.path.dirname(DATASET_ROOT)

extracted = {}


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
    
if __name__ == '__main__':
    gnet = GraspNet(DATASET_ROOT, camera='kinect', split='test')
    geometries = []
    
    """ Select the scene and annotation id."""
    annotation_id = 0
    scene_id = 100
    object_id = 9  # For scene 100: 9,11,20,29,30,41,48,52,58,62
    
    """ Load scene and object meshes """
    geometries.append(gnet.loadScenePointCloud(sceneId=scene_id, annId=annotation_id, camera='kinect'))
    # geometries += gnet.loadSceneModel(sceneId=scene_id, annId=annotation_id)

    """ Load explanations"""
    lime_explanations = o3d.io.read_point_cloud(os.path.join(GNET_ROOT, 'grasp_lime/visu', f'scene_0{scene_id}.ply'))
    lime_explanations.points, lime_explanations.colors = generate_grasp_views(np.asarray(lime_explanations.points), np.asarray(lime_explanations.colors))
    # geometries.append(lime_explanations)
    
    """ Run Eval """
    net, gg, end_points, grasps_per_object, best_grasp_per_object, logits, prediction, point_cloud = inference(object_id=object_id)
    grasp_array = np.array([grasp.numpy() for grasp in best_grasp_per_object.values()])
    # grasp_array = np.array([best_grasp_per_object[object_id].numpy()])
    
    """ Calculate pointnet feature-based metrics """
    fbi_explanations = tuple_to_pointcloud(get_fbi(end_points['fp2_xyz'], end_points['fp2_features'], type='sum', rm_outliers=True))
    geometries.append(fbi_explanations)
    
    """ Add selected grasps to plot """
    one_object_grasps = np.array([grasp.numpy() for grasp in grasps_per_object[object_id]])
    best_grasp_group = GraspGroup(grasp_array)
    one_object_group = GraspGroup(one_object_grasps)
    geometries += best_grasp_group.to_open3d_geometry_list()
    # geometries += one_object_group.to_open3d_geometry_list()

    o3d.visualization.draw_geometries(geometries)
    