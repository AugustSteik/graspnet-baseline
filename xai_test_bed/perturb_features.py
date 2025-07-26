import torch
import numpy as np
import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt


def perturb_features(end_points, original_fp2_featues, step=2, min=0, max=8):
    """
    Perturb the features of the input data by adding a small noise.

    Args:
        end_points (dict): Input data containing extracted point cloud features.
        perturbation (float): The amount of noise to add to the features.

    Returns:
        dict: The input data with perturbed features.
    """
    

    return end_points

def calculate_feature_metrics(end_points):
    """
    Visualise the features of the input data using Open3D.

    Args:
        end_points (dict): Input data containing point cloud features.
    """
    stats_dict = {}
    
    fp2_features = end_points['fp2_features'].squeeze().cpu().numpy()
    
    overall_stats = {
        "mean": np.mean(fp2_features),  # Mean of all features
        "variance": np.var(fp2_features),  # Variance of all features
        "min": np.min(fp2_features),  # Minimum value of all features
        "max": np.max(fp2_features),  # Maximum value of all features
    }

    # Calculate per-row statistics
    per_row_stats = {
        "means": np.mean(fp2_features, axis=1),  # Mean of each row
        "variances": np.var(fp2_features, axis=1),  # Variance of each row
        "mins": np.min(fp2_features, axis=1),  # Minimum value in each row
        "maxs": np.max(fp2_features, axis=1),  # Maximum value in each row
    }

    # Combine into a single dictionary
    stats_dict = {
        "overall": overall_stats,
        "per_row": per_row_stats,
    }

    return stats_dict
