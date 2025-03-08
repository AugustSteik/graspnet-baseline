"""PLOT EVERYTHING NICELY
This script is used to plot the point cloud, grasps, objects and any explanations in the scene in a nice way.
"""
import os
from graspnetAPI import GraspNet, Grasp
import open3d as o3d
import cv2
import sys
import torch


sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from grasp_lime.LIME_single_grasp import init_model, run_eval, detatch_dict


GNET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataset')
gnet = GraspNet(GNET_ROOT, camera='kinect', split='test')
extracted = {}

def extract_seed_points_and_scores(module, input, output):
    """Hook function to extract the seed points and objectness score from the model.
    """
    extracted.update(detatch_dict(output))

def check_objectness_scores(objectness_scores, threshold):
    """ Check the objectness probabilities."""
    probabilities = torch.sigmoid(objectness_scores.squeeze())
    grasp = probabilities[0, :][torch.where(probabilities[0, :] > threshold)]
    no_grasp = probabilities[1, :][torch.where(probabilities[1, :] >= 1- threshold)]
    objectness_pred = torch.argmax(objectness_scores, 1)
    print(len(grasp), len(no_grasp))
    return
    
if __name__ == '__main__':
    geometries = []
    """ Select the scene and annotation id."""
    annotation_id = 0
    scene_id = 0

    """ Load the scene and get grasp predictions."""
    scene = gnet.loadScenePointCloud(sceneId=scene_id, annId=annotation_id, camera='kinect')

    net, dataloader = init_model(wrapped=False, hook_fn=extract_seed_points_and_scores, scene_id=scene_id, image_range=(annotation_id, annotation_id+1), get_labels=True)
    _, predictions = run_eval(net, dataloader, hook_fn=extract_seed_points_and_scores, raw_predictions=False)

    check_objectness_scores(extracted['objectness_score'], 0.9)
    
    geometries.append(scene)
    seed_points_pc = o3d.geometry.PointCloud()
    seed_points_pc.points = o3d.utility.Vector3dVector(extracted['point_clouds'].squeeze().numpy())
    
    # geometries.append(seed_points_pc)
    # geometries += predictions.to_open3d_geometry_list()
    # o3d.visualization.draw_geometries(geometries)
    