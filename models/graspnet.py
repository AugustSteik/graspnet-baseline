""" GraspNet baseline model definition.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from models.backbone import Pointnet2Backbone
from models.modules import ApproachNet, CloudCrop, OperationNet, ToleranceNet
from models.loss import get_loss
from utils.loss_utils import GRASP_MAX_WIDTH, GRASP_MAX_TOLERANCE
from utils.label_generation import process_grasp_labels, match_grasp_view_and_label, batch_viewpoint_params_to_matrix

# My modules:
from models.modules import MLPApproachNet
from utils.label_generation import process_my_grasp_labels
from models.loss import get_approach_loss

class MyGraspNetStage1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300):
        super().__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.vpmodule = MLPApproachNet(num_view, seed_feature_dim=256)
        
    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)
        end_points = self.vpmodule(seed_xyz, seed_features, end_points)
        return end_points

class GraspNetStage1(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300):
        super().__init__()
        self.backbone = Pointnet2Backbone(input_feature_dim)
        self.vpmodule = ApproachNet(num_view, 256)

    def forward(self, end_points):
        pointcloud = end_points['point_clouds']
        seed_features, seed_xyz, end_points = self.backbone(pointcloud, end_points)
        end_points = self.vpmodule(seed_xyz, seed_features, end_points)
        return end_points


class GraspNetStage2(nn.Module):
    def __init__(self, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.num_angle = num_angle
        self.num_depth = num_depth
        self.is_training = is_training
        self.crop = CloudCrop(64, 3, cylinder_radius, hmin, hmax_list)
        self.operation = OperationNet(num_angle, num_depth)
        self.tolerance = ToleranceNet(num_angle, num_depth)
    
    def forward(self, end_points):
        pointcloud = end_points['input_xyz']
        if self.is_training:
            grasp_top_views_rot, _, _, _, end_points = match_grasp_view_and_label(end_points)
            seed_xyz = end_points['batch_grasp_point']
        else:
            grasp_top_views_rot = end_points['grasp_top_view_rot']
            seed_xyz = end_points['fp2_xyz']

        vp_features = self.crop(seed_xyz, pointcloud, grasp_top_views_rot)
        end_points = self.operation(vp_features, end_points)
        end_points = self.tolerance(vp_features, end_points)

        return end_points

class GraspNet(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = GraspNetStage1(input_feature_dim, num_view)
        self.grasp_generator = GraspNetStage2(num_angle, num_depth, cylinder_radius, hmin, hmax_list, is_training)

    def forward(self, end_points):
        end_points = self.view_estimator(end_points)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
        end_points = self.grasp_generator(end_points)
        return end_points

class MyGraspNet(nn.Module):
    def __init__(self, input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=True):
        super().__init__()
        self.is_training = is_training
        self.view_estimator = MyGraspNetStage1(input_feature_dim, num_view)
        self.grasp_generator = GraspNetStage2(num_angle, num_depth, cylinder_radius, hmin, hmax_list, is_training)

    def forward(self, end_points):
        end_points = self.view_estimator(end_points)
        if self.is_training:
            end_points = process_grasp_labels(end_points)
        end_points = self.grasp_generator(end_points)
        return end_points
    
def pred_decode(end_points):
    """

    Args:
        end_points (str: torch.Tensor([])): Predicted grasp parameters among other things.
        Sets any grasp_width values greater than GRASP_MAX_WIDTH to GRASP_MAX_WIDTH. 
        Obtains an in-plane grasp angle from the optimum predicted angle class for each point, for each approach distance.
        Gets the optimum approach distance for each point.
        
    Returns:
        grasp_preds (torch.Tensor([])): 
    """
    batch_size = len(end_points['point_clouds'])
    grasp_preds = []
    for i in range(batch_size):
        ## load predictions
        objectness_score = end_points['objectness_score'][i].float()
        grasp_score = end_points['grasp_score_pred'][i].float()
        grasp_center = end_points['fp2_xyz'][i].float()
        approaching = -end_points['grasp_top_view_xyz'][i].float()
        grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i]
        grasp_width = 1.2 * end_points['grasp_width_pred'][i]
        grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)
        grasp_tolerance = end_points['grasp_tolerance_pred'][i]

        ## slice preds by angle
        # grasp angle
        grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
        grasp_angle = grasp_angle_class.float() / 12 * np.pi
        # grasp score & width & tolerance
        grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
        grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)
        grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)
        grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_angle_class_).squeeze(0)

        ## slice preds by score/depth
        # grasp depth
        grasp_depth_class = torch.argmax(grasp_score, 1, keepdims=True)
        grasp_depth = (grasp_depth_class.float()+1) * 0.01
        # grasp score & angle & width & tolerance
        grasp_score = torch.gather(grasp_score, 1, grasp_depth_class)
        grasp_angle = torch.gather(grasp_angle, 1, grasp_depth_class)
        grasp_width = torch.gather(grasp_width, 1, grasp_depth_class)
        grasp_tolerance = torch.gather(grasp_tolerance, 1, grasp_depth_class)

        ## slice preds by objectness
        objectness_pred = torch.argmax(objectness_score, 0)
        objectness_mask = (objectness_pred==1)
        grasp_score = grasp_score[objectness_mask]
        grasp_width = grasp_width[objectness_mask]
        grasp_depth = grasp_depth[objectness_mask]
        approaching = approaching[objectness_mask]
        grasp_angle = grasp_angle[objectness_mask]
        grasp_center = grasp_center[objectness_mask]
        grasp_tolerance = grasp_tolerance[objectness_mask]
        grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

        ## convert to rotation matrix
        Ns = grasp_angle.size(0)
        approaching_ = approaching.view(Ns, 3)
        grasp_angle_ = grasp_angle.view(Ns)
        rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
        rotation_matrix = rotation_matrix.view(Ns, 9)

        # merge preds
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_preds.append(torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids], axis=-1))
    return grasp_preds

def pred_decode_per_object(end_points):
    """

    Args:
        end_points (str: torch.Tensor([])): Predicted grasp parameters among other things.
        Sets any grasp_width values greater than GRASP_MAX_WIDTH to GRASP_MAX_WIDTH. 
        Obtains an in-plane grasp angle from the optimum predicted angle class for each point, for each approach distance.
        Gets the optimum approach distance for each point.
        
    Returns:
        grasps_per_object (torch.Tensor([])): 
    """
    batch_size = len(end_points['point_clouds'])
    object_ids_per_point = -1 + torch.gather(end_points['objectness_label_with_object_ids'], 1, end_points['fp2_inds'].long()).squeeze().cpu().numpy()
    grasps_per_object = {int(k): [] for k in np.unique(object_ids_per_point)}
    for i in range(batch_size):
        for j, object_id in enumerate(object_ids_per_point):
            ## load predictions
            objectness_score = end_points['objectness_score'][i, :, j].float()
            grasp_score = end_points['grasp_score_pred'][i, :, j, :].float()
            grasp_center = end_points['fp2_xyz'][i, j, :].float()
            approaching = -end_points['grasp_top_view_xyz'][i, j, :].float()
            grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i, :, j, :]
            grasp_width = 1.2 * end_points['grasp_width_pred'][i, :, j, :]
            grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)
            grasp_tolerance = end_points['grasp_tolerance_pred'][i, :, j, :]
            

            ## slice preds by angle
            # grasp angle
            grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
            grasp_angle = grasp_angle_class.float() / 12 * np.pi
            # grasp score & width & tolerance
            grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
            grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)
            grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)
            grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_angle_class_).squeeze(0)

            ## slice preds by score/depth
            # grasp depth
            grasp_depth_class = torch.argmax(grasp_score, 0, keepdims=True)
            grasp_depth = (grasp_depth_class.float()+1) * 0.01
            # grasp score & angle & width & tolerance
            grasp_score = torch.gather(grasp_score, 0, grasp_depth_class)
            grasp_angle = torch.gather(grasp_angle, 0, grasp_depth_class)
            grasp_width = torch.gather(grasp_width, 0, grasp_depth_class)
            grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_depth_class)

            ## slice preds by objectness
            # objectness_pred = torch.argmax(objectness_score, 0)
            # objectness_mask = (objectness_pred==1)
            # grasp_score = grasp_score[objectness_mask]
            # grasp_width = grasp_width[objectness_mask]
            # grasp_depth = grasp_depth[objectness_mask]
            # approaching = approaching[objectness_mask]
            # grasp_angle = grasp_angle[objectness_mask]
            # grasp_center = grasp_center[objectness_mask]
            # grasp_tolerance = grasp_tolerance[objectness_mask]
            grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE

            ## convert to rotation matrix
            Ns = grasp_angle.size(0)  # Number of samples - Here is always 1
            approaching_ = approaching.view(Ns, 3)
            grasp_angle_ = grasp_angle.view(Ns)
            rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
            rotation_matrix = rotation_matrix.view(9)

            # merge preds
            grasp_height = 0.02 * torch.ones_like(grasp_score)
            obj_ids = -1 * torch.ones_like(grasp_score)
            grasps_per_object[object_id].append(torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids], axis=0).cpu())
    return grasps_per_object

def pred_decode_per_object_view_logits(end_points):
    """

    Args:
        end_points (str: torch.Tensor([])): Predicted grasp parameters among other things.
        Sets any grasp_width values greater than GRASP_MAX_WIDTH to GRASP_MAX_WIDTH. 
        Obtains an in-plane grasp angle from the optimum predicted angle class for each point, for each approach distance.
        Gets the optimum approach distance for each point.
        
    Returns:
        grasps_per_object (torch.Tensor([])): 
    """
    batch_size = len(end_points['point_clouds'])
    object_ids_per_point = -1 + torch.gather(end_points['objectness_label_with_object_ids'], 1, end_points['fp2_inds'].long()).squeeze().cpu().numpy()
    
    grasps_per_object = {int(k): [] for k in np.unique(object_ids_per_point)}
    view_logits_per_object = {int(k): [] for k in np.unique(object_ids_per_point)}
    best_grasp_view_logits_per_object = {int(k): [] for k in np.unique(object_ids_per_point)}
    scores_per_object = {int(k): [] for k in np.unique(object_ids_per_point)}
    scores_per_object = {int(k): [] for k in np.unique(object_ids_per_point)}
    point_idxs_per_obj = {int(k): [] for k in np.unique(object_ids_per_point)}
    best_grasp_per_object = None
    
    for i in range(batch_size):
        for j, object_id in enumerate(object_ids_per_point):
            if object_id == -1:
                continue
            ## load predictions
            objectness_score = end_points['objectness_score'][i, :, j].float()
            grasp_score = end_points['grasp_score_pred'][i, :, j, :].float()
            grasp_center = end_points['fp2_xyz'][i, j, :].float()
            approaching = -end_points['grasp_top_view_xyz'][i, j, :].float()
            grasp_angle_class_score = end_points['grasp_angle_cls_pred'][i, :, j, :]
            grasp_width = 1.2 * end_points['grasp_width_pred'][i, :, j, :]
            grasp_width = torch.clamp(grasp_width, min=0, max=GRASP_MAX_WIDTH)
            grasp_tolerance = end_points['grasp_tolerance_pred'][i, :, j, :]
            
            view_logits = end_points['view_score'][i, j, :]
            # top_view_ind = end_points['grasp_top_view_inds'][i, j]
            # top_view_logit = view_logits[top_view_ind.item()]
            # top_view_calc = torch.argmax(view_logits)

            ## slice preds by angle
            # grasp angle
            grasp_angle_class = torch.argmax(grasp_angle_class_score, 0)
            grasp_angle = grasp_angle_class.float() / 12 * np.pi
            # grasp score & width & tolerance
            grasp_angle_class_ = grasp_angle_class.unsqueeze(0)
            grasp_score = torch.gather(grasp_score, 0, grasp_angle_class_).squeeze(0)
            grasp_width = torch.gather(grasp_width, 0, grasp_angle_class_).squeeze(0)
            grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_angle_class_).squeeze(0)

            ## slice preds by score/depth
            # grasp depth
            grasp_depth_class = torch.argmax(grasp_score, 0, keepdims=True)
            grasp_depth = (grasp_depth_class.float()+1) * 0.01
            # grasp score & angle & width & tolerance
            grasp_score = torch.gather(grasp_score, 0, grasp_depth_class)
            grasp_angle = torch.gather(grasp_angle, 0, grasp_depth_class)
            grasp_width = torch.gather(grasp_width, 0, grasp_depth_class)
            grasp_tolerance = torch.gather(grasp_tolerance, 0, grasp_depth_class)
            
            grasp_score = grasp_score * grasp_tolerance / GRASP_MAX_TOLERANCE
            scores_per_object[object_id].append(grasp_score)
            point_idxs_per_obj[object_id].append((j, grasp_score))
            
            ## convert to rotation matrix
            Ns = grasp_angle.size(0)  # Number of samples - Here is always 1
            approaching_ = approaching.view(Ns, 3)
            grasp_angle_ = grasp_angle.view(Ns)
            rotation_matrix = batch_viewpoint_params_to_matrix(approaching_, grasp_angle_)
            rotation_matrix = rotation_matrix.view(9)

            # merge preds
            grasp_height = 0.02 * torch.ones_like(grasp_score)
            obj_ids = -1 * torch.ones_like(grasp_score)
            grasps_per_object[object_id].append(torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, rotation_matrix, grasp_center, obj_ids], axis=0).cpu()) # type: ignore
            view_logits_per_object[object_id].append(view_logits)
            
    best_grasp_per_object = get_best_grasp_per_object(grasps_per_object)
    best_grasp_view_logits_per_object = get_best_grasp_view_logits(view_logits_per_object, scores_per_object)

    return grasps_per_object, best_grasp_per_object, view_logits_per_object, best_grasp_view_logits_per_object, point_idxs_per_obj

def get_best_grasp_per_object(grouped_grasps):
    top_grasps = {}
    for object_id, grasps in grouped_grasps.items():
        if object_id == -1:
            continue
        scores = torch.tensor([grasp[0] for grasp in grasps])
        top_grasps[object_id] = grasps[torch.argmax(scores)]
    return top_grasps

def get_best_grasp_view_logits(grouped_view_logits, scores_per_object):
    top_grasp_view_logits = {}
    for object_id in grouped_view_logits.keys():
        if object_id == -1:
            continue
        best_grasp_idx = torch.argmax(torch.cat(scores_per_object[object_id]))
        top_grasp_view_logits[object_id] = grouped_view_logits[object_id][best_grasp_idx]
    return top_grasp_view_logits
