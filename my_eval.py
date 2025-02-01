# from unittest.mock import MagicMock
# import sys
# # Spoof warnings
# sys.modules["geometry_msgs"] = MagicMock()
# sys.modules["rospy"] = MagicMock()
# sys.modules["rosbag"] = MagicMock()
# sys.modules["sensor_msgs"] = MagicMock()
# sys.modules["autolab_core"] = MagicMock()
# sys.modules["autolab_core.rigid_transformations"] = MagicMock()

import graspnetAPI as api
import numpy as np
from graspnetAPI import GraspNet
from custom_eval import GraspNetEval
import open3d as o3d
import cv2
import os
from random import randint


def visualise_grasps(pred_root, scene_ids, ann_ids, show_grasp_points=True):
    graspnet_root = 'dataset'
    camera = 'kinect'
    pred_paths = {}
    for scene_id in scene_ids:
        grasps_list = []
        for ann_id in ann_ids:
            pred_path = os.path.join(pred_root, f'scene_{str(scene_id).zfill(4)}',camera, f'{str(ann_id).zfill(4)}.npy')
            grasps = api.grasp.GraspGroup(np.load(pred_path)).sort_by_score()
            grasps_list.append(grasps)
        pred_paths[f'{str(scene_id).zfill(4)}'] = grasps_list
            
    # Initialise a GraspNet instance
    g=GraspNet(graspnet_root, camera=camera, split='custom', sceneIds=scene_ids)

    view_annotations = {}
    
    for k, v in pred_paths.items():  # scene_id, [grasps, grasps] (for each annotation id)
        geometries=[]
        scene_geometries = []
        for idx, grasps in enumerate(v):
            
            ann_id = ann_ids[idx]
            geometries.append(g.loadScenePointCloud(sceneId=int(k), annId=ann_id, camera='kinect'))  # Load scene pc for each annotation id
            geometries += grasps.to_open3d_geometry_list()
            
            if show_grasp_points:
                for i in range(len(grasps)):  # for grasp object in grasp group object
                    grasp = grasps.__getitem__(i)
                    annotation = grasp.translation
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                    sphere.translate(annotation)
                    sphere.paint_uniform_color([1, 0, 0])  # Red for grasp point markers
                    geometries.append(sphere)
            
            scene_geometries.append(geometries)
            
        view_annotations[k] = scene_geometries
                
    for k, v in view_annotations.items():  # for scene_id: [geometries_list, geometries_list]
        print(k)
        for geometries in v:
            
            o3d.visualization.draw_geometries(geometries)
    return

def visualise_object_grasps(pred_path, object_ids):
    camera = 'kinect'
    graspnet_root = 'dataset'
    
    models_root = os.path.join(graspnet_root, 'models')
    
    for object in object_ids:
        geometries = []
        
        model_path = os.path.join(models_root, str(object).zfill(3), 'textured.obj')
        grasp_path = os.path.join(pred_path, str(object), camera, f'{str(object).zfill(4)}.npy')
        
        mesh = o3d.io.read_triangle_mesh(model_path)
        mesh.compute_vertex_normals()  # Enable texture rendering
        geometries.append(mesh)
        
        grasps = api.grasp.GraspGroup(np.load(grasp_path)).sort_by_score()
        print(grasps[:5])
        print(object)
        geometries += grasps.to_open3d_geometry_list()[:2]
        
        o3d.visualization.draw_geometries(geometries)
            
def evaluate(dump_dir, scene_ids):
    gnet_root = 'dataset'
    ge = GraspNetEval(gnet_root, camera='kinect', split='custom')
    res, ap = ge.eval_custom(scene_ids, dump_dir)
    save_dir = os.path.join(dump_dir, 'ap_{}.npy'.format('kinect'))
    np.save(save_dir, res)
    return
            
def evaluate_from_np(pred_path):
    camera = 'kinect'
    threshold = 1e-8
    
    aps = np.load(os.path.join(pred_path, 'ap_kinect.npy'))
    mask = np.abs(aps) > threshold
    restored_aps = np.where(mask, aps, np.nan)  # Retains the 4D structure

    ap = [
        np.nanmean(restored_aps),
        np.nanmean(restored_aps[0:30]),  # Seen
        np.nanmean(restored_aps[30:60]),  # Similar
        np.nanmean(restored_aps[60:90])  # Novel
    ]
    print('\nEvaluation Result:\n----------\n{}, AP={}, AP Seen={}, AP Similar={}, AP Novel={}'.format(camera, ap[0], ap[1], ap[2], ap[3]))
    return    
    
if __name__ == '__main__':
    
    pred_path = 'logs\\my_logs\\train_3_mlp_approachnet_full\\test_outputs'
    pc_pred_path = 'C:\\Development\\GraspNet\\graspnet-baseline\\logs\\my_logs\\train_3_mlp_approachnet_full\\model_pc_outputs'
    
    num_scenes = 1
    seen = [randint(100, 130) for _ in range(num_scenes)]
    similar = [randint(130, 160) for _ in range(num_scenes)]
    novel = [randint(160, 190) for _ in range(num_scenes)]
    
    scene_ids = seen + similar + novel
    ann_ids = [0]  # object 6d pose annotations for each scene
    
    object_ids = [randint(0, 88)]
    
    # visualise_grasps(pred_path, scene_ids, ann_ids)
    visualise_object_grasps(pc_pred_path, object_ids)
    # evaluate(pred_path, list ( range(100, 190) ))
    # evaluate_from_np(pred_path)
    