import matplotlib.pyplot as plt
import sys, os
import torch
import copy
import pandas as pd
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

GNET_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(GNET_ROOT)
ROOT = os.path.dirname(__file__)
sys.path.append(ROOT)

from models.loss import compute_objectness_loss, compute_view_loss, compute_grasp_loss  # grasp loss is multiplied by 0.2 for the overall loss sum

from xai_inference_utils import load_data, load_model, inference_one_batch, check_uniqueness, add_labels
from perturb_features import perturb_features, calculate_feature_metrics


class FeatureGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        
        # Hook the last conv layer
        self.model.conv3.register_forward_hook(self.save_activations)
        self.model.conv3.register_full_backward_hook(self.save_gradients)
        
    def save_activations(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def forward(self, seed_xyz, seed_features, end_points):
        return self.model(seed_xyz, seed_features, end_points)

def compute_all_feature_importances(model, seed_xyz, seed_features, end_points):
    """
    Compute feature importance for all seed points.
    Returns:
        objectness_importances: (num_seed, feature_dim)
        view_importances: (num_seed, feature_dim)
    """
    model.eval()
    seed_features = seed_features.clone().detach().requires_grad_(True)
    
    # Forward pass
    end_points = model(seed_xyz, seed_features, end_points)
    num_seed = seed_xyz.size(1)
    feature_dim = seed_features.size(1)
    
    objectness_importances = np.zeros((num_seed, feature_dim))
    view_importances = np.zeros((num_seed, feature_dim))
    
    for seed_idx in range(num_seed):
        # --- Objectness importance ---
        model.zero_grad()
        target_score = end_points['objectness_score'][0, 1, seed_idx]  # Class 1 (graspable)
        target_score.backward(retain_graph=True)
        grad_objectness = seed_features.grad[0, :, seed_idx].abs().clone().detach().cpu().numpy()
        objectness_importances[seed_idx] = grad_objectness
        
        # --- View importance ---
        model.zero_grad()
        view_score = end_points['view_score'][0, seed_idx, :]
        target_view_idx = torch.argmax(view_score).item()
        target_score = view_score[target_view_idx]
        target_score.backward(retain_graph=True)
        grad_view = seed_features.grad[0, :, seed_idx].abs().clone().detach().cpu().numpy()
        view_importances[seed_idx] = grad_view
        
        # Reset gradients
        model.zero_grad()
        seed_features.grad.zero_()
    
    return objectness_importances, view_importances

def aggregate_to_original_points(importances, seed_inds, num_original_points):
    """
    Args:
        importances: (num_seed, feature_dim)
        seed_inds: (num_seed,) indices of seeds in the original point cloud
    Returns:
        original_importances: (num_original_points, feature_dim)
    """
    original_importances = np.zeros((num_original_points, importances.shape[1]))
    for seed_idx in range(importances.shape[0]):
        original_idx = seed_inds[seed_idx]
        original_importances[original_idx] += importances[seed_idx]
    return original_importances

def visualize_heatmap(original_xyz, scores, title):
    """
    Args:
        original_xyz: (num_original_points, 3) coordinates of the original points
        scores: (num_original_points,) importance scores
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        original_xyz[:, 0], original_xyz[:, 1], original_xyz[:, 2],
        c=scores, cmap='jet', s=10, alpha=0.7
    )
    plt.colorbar(sc)
    plt.title(title)
    plt.show()

def generate_grad_cam_explanations():
    data_path = f'{ROOT}/scene_100-end_points.json'  # pointnetbackbone-end_points_full.json
    
    approachnet = load_model()
    end_points = load_data(data_path)
    seed_xyz = end_points['fp2_xyz']
    seed_features = end_points['fp2_features']
    # gradcam = FeatureGradCAM(approachnet)
    # seed_inds = end_points['fp2_inds'][0].cpu().numpy()  # (num_seed,)
    # num_original_points = 20_000  # Total points in the original input  
    
    objectness_importances, view_importances = compute_all_feature_importances(approachnet, seed_xyz, seed_features, end_points)
    point_scores_objectness = objectness_importances.sum(axis=1)
    point_scores_view = view_importances.sum(axis=1)
    
    return point_scores_objectness, point_scores_view
    
    
if __name__ == '__main__':
    

    # Example usage:
    # Assuming `end_points` contains 'seed_inds' (indices of seeds in the original cloud)

    # Aggregate objectness and view importances
    # original_objectness = aggregate_to_original_points(objectness_importances, seed_inds, num_original_points)
    # original_view = aggregate_to_original_points(view_importances, seed_inds, num_original_points)
    
    # Sum across feature dimensions


    # Alternatively, use max or mean:
    # point_scores_objectness = original_objectness.max(axis=1)
    
    point_scores_objectness, _ = generate_grad_cam_explanations()
    # Example usage:
    visualize_heatmap(seed_xyz.cpu().squeeze(), point_scores_objectness, "Objectness Importance")
    # visualize_heatmap(original_xyz, point_scores_objectness, "Objectness Importance")
    # visualize_heatmap(original_xyz, point_scores_view, "View Importance")


