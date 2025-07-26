import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
from matplotlib.widgets import Slider
from plotly.subplots import make_subplots
import torch
import open3d as o3d
import json
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

#Successfully uninstalled numpy-1.20.3

run_name = 'run_5'
scene_id = '100'

DATA_PATH = os.path.join(os.path.dirname(__file__), scene_id, run_name)
SAVE_DIR = os.path.join(DATA_PATH, 'analysis')

base_case = np.loadtxt(
    os.path.join(DATA_PATH, 'base_case.csv'),
    delimiter=",",
    skiprows=1
)
full_run = np.loadtxt(
    os.path.join(DATA_PATH, 'full_run.csv'),
    delimiter=",",
    skiprows=1
)

testing_corrcoeffs = np.array([
 -0.01895815, -0.01980727, -0.01884496,  0.00202005, -0.00593579,  0.0109335,
 -0.00073236, -0.00867436,  0.0091911 ,  0.04406265,  0.02315794,  0.00890769,
  0.00942792, -0.00185504,  0.00755731,  0.00869704, -0.00397496, -0.009071, 
  0.0177273 ,  0.02114388,  0.0064947 ,  0.03593919,  0.04310052,  0.03675457,
  0.01823843,  0.00768403, -0.00366871,  0.01496729,  0.02169225,  0.01320403,
  0.00772489,  0.02551947,  0.01668244,  0.00853856,  0.01027857,  0.01989309,
  0.01760672, -0.00145968,  0.00388894,  0.04532169,  0.03714248,  0.03287609,
  0.02008745,  0.0206108 ,  0.03683131,  0.03971052,  0.04753301,  0.04021289,
  0.04838758,  0.03558287,  0.02896434,  0.02588141,  0.0169215 ,  0.01786084,
  0.03415433,  0.04032305,  0.04769547,  0.038463  ,  0.02588543,  0.02423179,
  0.01860833,  0.02145918,  0.04961227,  0.04740716,  0.03948815,  0.05691298,
  0.05971888,  0.05147914,  0.05413269,  0.05806591,  0.0557274 ,  0.0521578,
  0.0544487 ,  0.05712568,  0.04956513,  0.04793305,  0.04780819,  0.05038876,
  0.03625728,  0.03237176,  0.01054589,  0.01615445, -0.00543755,  0.01263942,
  0.01720723,  0.00136135,  0.00865394,  0.02394631,  0.01108563,  0.04997312,
  0.04601528,  0.04227037,  0.04140678,  0.04813635,  0.05198451,  0.05512819,
  0.03203799,  0.02699415,  0.02191313,  0.039715  ,  0.04519589,  0.04217976,
  0.04931829,  0.04831781,  0.04654098,  0.05350156,  0.05204027,  0.05788933,
  0.06361771,  0.05364242,  0.05896998,  0.06524569,  0.07094452,  0.06960134,
  0.07152059,  0.06640879,  0.05662814,  0.06744579,  0.06948362,  0.07409004,
  0.07028736,  0.07230709,  0.07074112,  0.05817366,  0.05644342,  0.05948873,
  0.05746339,  0.05890492,  0.0699369 ,  0.06007712,  0.06098707,  0.06002735,
  0.06159479,  0.05208049,  0.04199628,  0.01415141,  0.0116605 ,  0.00957068,
  0.00528853, -0.01813957, -0.00754389,  0.00083538,  0.01722387,  0.03541359,
  0.00812092, -0.01524854, -0.01266029, -0.00812992,  0.00306967, -0.00125712,
  0.01199568,  0.01664643,  0.00799557,  0.00228936, -0.03258374, -0.04531358,
 -0.02728603, -0.03975788, -0.03767898, -0.03464141, -0.02183838, -0.01156177,
 -0.02222321, -0.00853305,  0.01071167, -0.00052562, -0.00245951, -0.00504734,
 -0.02322204, -0.0456733 , -0.04013078, -0.05221321, -0.0702941 , -0.07347384,
 -0.06536261, -0.06614852, -0.06374022, -0.06144096, -0.05399733, -0.05699442,
 -0.06089274, -0.0682311 , -0.06523315, -0.06968256, -0.05991635, -0.0530153, 
 -0.05361457, -0.06774402, -0.02908806, -0.00314772, -0.02672617, -0.02658975,
  0.01633551, -0.00268242, -0.00287933,  0.01357789,  0.03421876,  0.03613611,
  0.01072702,  0.03375327,  0.02198278,  0.02183128,  0.02867867,  0.01261344,
  0.01203298,  0.0156626 ,  0.02271051,  0.00156719, -0.00955767, -0.02644541,
 -0.0387193 , -0.03688696, -0.01428398, -0.03841942, -0.03768515, -0.04146031,
 -0.04802135, -0.01743107, -0.02258834, -0.02190181, -0.00654473,  0.00447722,
  0.00950103,  0.04442518,  0.04013382,  0.02737195,  0.03772726,  0.02436185,
  0.02219773,  0.03082678,  0.02147056,  0.01416101,  0.01138857,  0.00141846,
  0.00172332, -0.03423006, -0.0184817 , -0.00622967, -0.01533665, -0.02547195,
 -0.04239899, -0.05083552, -0.05272892, -0.04014462, -0.05939079, -0.08406001,
 -0.0866103 , -0.07535371, -0.07531424, -0.0684035 , -0.05885806, -0.05146297,
 -0.04761856, -0.04023051, -0.03345952, -0.01547918
 ])

corrcoef_keys = ['objectness_score_corrcoef', 
                    'top_grasp_score_corrcoef', 
                    'base_case_top_view_ind_grasp_score_corrcoef', 
                    'top_view_ind_corrcoef', 
                    'original_view_grasp_score_change_corrcoef']
n_top = 25

#                                                           full_run.csv
#      0           1              2          3                     4                       5                         6
# ['point_idx', 'feature_idx', 'value', 'objectness_score', 'grasp_top_view_ind', 'grasp_top_view_score', 'grasp_base_top_view_ind_score']

def calculate_metrics(data: np.ndarray=full_run, kwargs: dict={'original_data': base_case, 'masked': False}) -> dict:
    num_features = int(np.max(data[:, 1])) + 1  # Number of unique features
    original_data = kwargs['original_data']
    
    if kwargs['masked']:
        original_data_objectness_mask = (original_data[:, 1] > 0.1)
        original_data = original_data[original_data_objectness_mask, :]
        objectness_points = (original_data[:, 0])  # get point ids where objectness is 1
        objectness_mask = np.isin(data[:, 0], objectness_points)
        data = data[objectness_mask, :]
        points = objectness_points.astype(int)
    else:
        points = np.array(range(int(np.max(data[:, 0])) + 1))
    
    out_dict = {str(i): {} for i in points}
    out_dict['params'] = {
        'num_points': len(points),
        'num_features': num_features,
        'valid_points': points,
    }
    
    point_keys = [str(i) for i in out_dict['params']['valid_points']]
    
    out_dict['global'] = {key: np.zeros(num_features, dtype=float) for key in corrcoef_keys}
    out_dict['consensus'] = {key:{'pos': np.zeros(num_features, dtype=float), 
                                  'neg': np.zeros(num_features, dtype=float)}  for key in corrcoef_keys}
    
    base_case_top_view_score = original_data[:, -1]  # base_case grasp_top_view_score, index 260
    for idx, point_idx in enumerate(points):
        point_mask = (data[:, 0] == point_idx)
        masked_data = data[point_mask, :]
        
        num_samples = masked_data.shape[0]  # Number of perturbations for this point
        if num_samples == 0:
            continue
        if 'num_samples_per_point' not in out_dict['params']:
            out_dict['params']['num_samples_per_point'] = num_samples
        features = np.zeros((num_samples, num_features), dtype=float)
        objectness_score = masked_data[:, 3] - base_case[point_idx, -3]  # objectness_score TODO: check this subtraction of the original score
        top_grasp_score = masked_data[:, 5]  # grasp_top_view_score
        grasp_base_top_view_ind_score = masked_data[:, 6]  # grasp_base_top_view_ind_score
        grasp_top_view_ind = masked_data[:, 4]  # grasp_top_view_inds
        
        impact_on_original_view_score = grasp_base_top_view_ind_score - base_case_top_view_score[idx]
        
        # Fill feature matrix
        feature_idxs = masked_data[:, 1].astype(int)
        feature_values = masked_data[:, 2] 
        original_data_values = original_data[idx, feature_idxs + 2]
        features[np.arange(num_samples), feature_idxs] = feature_values - original_data_values  # subtract original value (1st 2 columns are point_idx and objectness_label)
        
        # Compute correlation
        # Transpose to get features as rows (variables) and samples as columns (observations)
        o_data_matrix = np.vstack((features.T, objectness_score))  # type: ignore
        o_corr_matrix = np.corrcoef(o_data_matrix)
        # num_features = features.shape[1]
        # spearman_corrs = []

        # for i in range(num_features):
        #     corr, _ = spearmanr(features[:, i], objectness_score)
        #     spearman_corrs.append(corr)

        # # Convert to a NumPy array if needed
        # o_corr_matrix = np.array(spearman_corrs)

        tg_data_matrix = np.vstack((features.T, top_grasp_score))  # type: ignore
        tg_corr_matrix = np.corrcoef(tg_data_matrix)
        
        gb_data_matrix = np.vstack((features.T, grasp_base_top_view_ind_score))  # type: ignore
        gb_corr_matrix = np.corrcoef(gb_data_matrix)
        
        tv_data_matrix = np.vstack((features.T, grasp_top_view_ind))  # type: ignore
        tv_corr_matrix = np.corrcoef(tv_data_matrix)
        
        i_data_matrix = np.vstack((features.T, impact_on_original_view_score))  # type: ignore
        i_corr_matrix = np.corrcoef(i_data_matrix)
        
        # Extract correlations between each feature and target
        objectness_score_corrcoef = o_corr_matrix[-1, :-1]  # get last row, excluding the last column (as this is the correlation of the target with itself)
        # objectness_score_corrcoef = o_corr_matrix  # FROM SPEARMANSR TESTING
        top_grasp_score_corrcoef = tg_corr_matrix[-1, :-1]
        base_case_top_view_ind_grasp_score_corrcoef = gb_corr_matrix[-1, :-1]
        top_view_ind_corrcoef = tv_corr_matrix[-1, :-1]
        original_view_grasp_score_change_corrcoef = i_corr_matrix[-1, :-1]
        
        out_dict[str(point_idx)]['objectness_score_corrcoef'] = objectness_score_corrcoef
        out_dict[str(point_idx)]['top_grasp_score_corrcoef'] = top_grasp_score_corrcoef
        out_dict[str(point_idx)]['base_case_top_view_ind_grasp_score_corrcoef'] = base_case_top_view_ind_grasp_score_corrcoef
        out_dict[str(point_idx)]['top_view_ind_corrcoef'] = top_view_ind_corrcoef
        out_dict[str(point_idx)]['original_view_grasp_score_change_corrcoef'] = original_view_grasp_score_change_corrcoef
        
        for key in corrcoef_keys:
            sorted_indices = np.argsort(out_dict[str(point_idx)][key])
            # Top positive (highest positive correlations)
            top_positive_corrcoef = sorted_indices[-n_top:][::-1]  # Descending order
            out_dict[str(point_idx)][f'top_positive_{key}'] = {
                'indices': top_positive_corrcoef.tolist(),
                'values': out_dict[str(point_idx)][key][top_positive_corrcoef].tolist()
            }
            # Top negative (most negative correlations)
            top_negative_corrcoef = sorted_indices[:n_top]  # Already in ascending order
            out_dict[str(point_idx)][f'top_negative_{key}'] = {
                'indices': top_negative_corrcoef.tolist(),
                'values': out_dict[str(point_idx)][key][top_negative_corrcoef].tolist()
            }
    out_dict['global']['objectness_score_corrcoef'] = np.mean([out_dict[str(i)]['objectness_score_corrcoef'] for i in points], axis=0)
    out_dict['global']['top_grasp_score_corrcoef'] = np.mean([out_dict[str(i)]['top_grasp_score_corrcoef'] for i in points], axis=0)
    out_dict['global']['base_case_top_view_ind_grasp_score_corrcoef'] = np.mean([out_dict[str(i)]['base_case_top_view_ind_grasp_score_corrcoef'] for i in points], axis=0)
    out_dict['global']['top_view_ind_corrcoef'] = np.mean([out_dict[str(i)]['top_view_ind_corrcoef'] for i in points], axis=0)
    out_dict['global']['original_view_grasp_score_change_corrcoef'] = np.mean([out_dict[str(i)]['original_view_grasp_score_change_corrcoef'] for i in points], axis=0)
    
    # Get top positive and negative correlations for each metric
    for key in corrcoef_keys:
        sorted_indices = np.argsort(out_dict['global'][key])
        # Top positive (highest positive correlations)
        top_positive_corrcoef = sorted_indices[-n_top:][::-1]  # Descending orderhow to 
        out_dict['global'][f'top_positive_{key}'] = {
            'indices': top_positive_corrcoef.tolist(),
            'values': out_dict['global'][key][top_positive_corrcoef].tolist()
        }
        # Top negative (most negative correlations)
        top_negative_corrcoef = sorted_indices[:n_top]  # Already in ascending order
        out_dict['global'][f'top_negative_{key}'] = {
            'indices': top_negative_corrcoef.tolist(),
            'values': out_dict['global'][key][top_negative_corrcoef].tolist()
        }
    
    # Calculate consensus for features in top positive and negative correlations
    num_points = out_dict['params']['num_points']
    pos_count = np.zeros(num_features, dtype=int)
    neg_count = np.zeros(num_features, dtype=int)
    for key in corrcoef_keys:
        for point_key in point_keys:
            pos_count[np.array(out_dict[point_key][f'top_positive_{key}']['indices'])] += 1
            neg_count[np.array(out_dict[point_key][f'top_negative_{key}']['indices'])] += 1
        out_dict['consensus'][key]['pos'][:] = pos_count / num_points
        out_dict['consensus'][key]['neg'][:] = neg_count / num_points

        pos_count[:] = 0
        neg_count[:] = 0
    return out_dict
    
def plot_correlations(out_dict, key, show=True, save=False):
    """Plot top positive/negative correlations for a specific point."""
    plt.ioff()
    pos_data = out_dict['global'][f'top_positive_{key}']
    neg_data = out_dict['global'][f'top_negative_{key}']
    
    n_top = len(pos_data['indices'])
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10))
    corref_name = format_title(key, no_corref=True)
    titlee = f'Top Feature Correlations for {corref_name}, Globally'
    fig.suptitle(titlee, fontsize=16)
    
    # Positive correlations plot
    sns.barplot(y=pos_data['values'], x=pos_data['indices'], ax=ax1, color='skyblue')
    ax1.set_title(f'Top {n_top} Positive Correlations')
    ax1.set_ylabel('Correlation Coefficient')
    ax1.set_xlabel('Feature Index')
    
    # Negative correlations plot
    sns.barplot(y=neg_data['values'], x=neg_data['indices'], ax=ax2, color='salmon')
    ax2.set_title(f'Top {n_top} Negative Correlations')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_xlabel('Feature Index')

    if show:
        plt.show(block=True)  # block=True keeps the plot open
    if save:
        plt.savefig(f'{SAVE_DIR}/{titlee}.png')
        
    return

def plot_global_correlations(out_dict, key, show=True, save=False):
    """Plot top positive/negative correlations for a specific point."""
    plt.ioff()
    data = out_dict['global'][key]
    x_lims = list(range(0, len(data)))
    
    plt.figure(figsize=(12, 6))
    # Positive correlations plot
    sns.barplot(y=data, x=x_lims, color='skyblue')
    titlee = f'Mean {format_title(key)}'
    plt.title(titlee, size=22)
    plt.ylabel('Correlation Coefficient', size=20)
    plt.xlabel('Feature Index', size=20)
    
    max_features = len(data)
    # label_step = max(1, max_features // 30)
    label_step = 10
    xticks = np.arange(0, max_features, label_step)
    plt.xticks(xticks + 0.5, rotation=45)
    plt.xticks(size = 16)
    plt.yticks(size = 16)
    
    plt.tight_layout()  # Automatically adjust subplot margins
    plt.subplots_adjust(bottom=0.2)  # Extra space for x-axis label
    
    if show:
        plt.show(block=True)  # block=True keeps the plot open
    if save:
        plt.savefig(f'{SAVE_DIR}/{titlee}.png')
    return

def generate_explanations(seed_features, out_dict, key, count=False, global_calc=False, mask_features=False):
    """
    Generate explanations based on feature correlations and seed features.
    Returns explanations in range [-1, 1], where:
        - Positive values = positive correlation
        - Negative values = negative correlations
    """
    seed_features = seed_features.T  # gets it to: (1024, 256)

    explanations = np.zeros(seed_features.shape[0])
    points = out_dict['params']['valid_points']
    
    if mask_features:
        # consensus_thresh = 0.5
        feature_mask = np.array([False for _ in range(out_dict['params']['num_features'])], dtype=bool)
        feature_mask[out_dict['global'][f'top_positive_{key}']['indices']] = True
        feature_mask[out_dict['global'][f'top_negative_{key}']['indices']] = True
        # feature_mask = np.where(out_dict['consensus'][key]['pos'] >= consensus_thresh, True, feature_mask)
        # feature_mask = np.where(out_dict['consensus'][key]['neg'] >= consensus_thresh, True, feature_mask)
    else:
        feature_mask = np.array([True for _ in range(out_dict['params']['num_features'])], dtype=bool)
        
    if not count:
        for i in points:
            if not global_calc:
                weights = out_dict[str(i)][key]
            else:
                weights = out_dict['global'][key]
                
            weights = weights / np.max(np.abs(weights))  # Scale to [-1, 1]
            weighted_features = seed_features[i, :] * weights * feature_mask
            
            explanations[i] = np.sum(weighted_features)
    else:
        temp_feature = np.zeros(seed_features.shape[1], dtype=int)
        for i in points:
            temp_feature = np.where(out_dict['consensus'][key]['pos'] >= 0.99, 1, temp_feature)
            temp_feature = np.where(out_dict['consensus'][key]['neg'] >= 0.99, -1, temp_feature)
            explanations[i] = sum(temp_feature)
            temp_feature[:] = 0
            
    # explanations = explanations / np.max(np.abs(explanations))  # TODO check the effect of normalising at the end
    return explanations

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

# def generate_pc_data(seed_xyz, explanations):
#     """
#     Map normalized explanations ([-1, 1]) to a diverging color gradient:
#         - Red (negative correlation)
#         - White/Gray (neutral)
#         - Blue (positive correlation)
#     """
#     pc = o3d.geometry.PointCloud()
#     pc.points = o3d.utility.Vector3dVector(seed_xyz)
#     colours = np.zeros((len(explanations), 3))  # RGB array
    
#     for idx, val in enumerate(explanations):
#         if val < 0:  # Negative correlation → red
#             colours[idx, 0] = -val  # Red increases as val goes from 0 → -1
#             colours[idx, 2] = 0     # No blue
#         else:  # Positive correlation → blue
#             colours[idx, 2] = val    # Blue increases as val goes from 0 → 1
#             colours[idx, 0] = 0      # No red
    
#     pc.colors = o3d.utility.Vector3dVector(colours)
#     return pc
def generate_pc_data(seed_xyz, explanations, valid_points, asym_norm=True):
    """
    Normalize positive and negative values separately, then map to RGB:
    - Red for strong negative values
    - Blue for strong positive values
    - Gray/white near zero
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(seed_xyz[valid_points])
    colours = np.zeros((len(explanations), 3))  # RGB array

    if asym_norm:
        # Separate normalization for positive and negative sides
        max_pos = np.max(explanations)
        max_neg = np.min(explanations)  # [explanations < 0], initial=-1)

        for idx, val in enumerate(explanations):
            if val < 0:
                norm_val = val / abs(max_neg)  # Normalize to [-1, 0]
                colours[idx, 0] = -norm_val    # Red
            elif val > 0:
                norm_val = val / max_pos       # Normalize to [0, 1]
                colours[idx, 2] = norm_val     # Blue
            # else leave at zero for neutral (gray)
    else:
        for idx, val in enumerate(explanations):
            if val < 0:  # Negative correlation → red
                colours[idx, 0] = -val  # Red increases as val goes from 0 → -1
                colours[idx, 2] = 0     # No blue
            else:  # Positive correlation → blue
                colours[idx, 2] = val    # Blue increases as val goes from 0 → 1
                colours[idx, 0] = 0      # No red
    pc.colors = o3d.utility.Vector3dVector(colours[valid_points])
    return pc



def plot_consensus_bars(out_dict, key, show=True, save=False):
    """Plot top positive/negative correlations for a specific point."""
    plt.ioff()
    
    positive_consensus = out_dict['consensus'][key]['pos']
    negative_consensus = out_dict['consensus'][key]['neg']
    
    pos_indices = np.where(positive_consensus > 0)[0]
    pos_values = positive_consensus[positive_consensus > 0]

    # Get indices and values for negative consensus
    neg_indices = np.where(negative_consensus > 0)[0]
    neg_values = negative_consensus[negative_consensus > 0]

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Plot positive values in blue (default)
    plt.barh(pos_indices, pos_values, color='blue', label='Positive Consensus')

    # Plot negative values in red
    plt.barh(neg_indices, neg_values, color='red', label='Negative Consensus')
    
    # Customize the plot
    titlee = f'Consensus Data for Top Positive and Negative Correlations ({key})'
    plt.title(titlee)
    plt.ylabel('Feature Index')
    plt.xlabel('Consensus')

    # Combine all indices that have either positive or negative values
    all_indices = sorted(set(pos_indices).union(set(neg_indices)))
    plt.yticks(all_indices)  # Only show relevant indices
    
    # Add legend
    plt.legend()
    if show:
        plt.show(block=True)  # block=True keeps the plot open
    if save:
        plt.savefig(f'{SAVE_DIR}/{titlee}.png')
        
    return    
    
def plot_consensus_heatmap(out_dict, key, show=False, save=True):
    """Heatmap with perfectly positioned dual colorbars."""
    plt.ioff()
    
    positive_consensus = out_dict['consensus'][key]['pos']
    negative_consensus = out_dict['consensus'][key]['neg']
    
    # Prepare data
    pos_data = np.where(positive_consensus > 0, positive_consensus, np.nan)
    neg_data = np.where(negative_consensus > 0, negative_consensus, np.nan)
    
    # --- Setup figure with correct margins ---
    fig = plt.figure(figsize=(18, 5))
    ax = fig.add_axes([0.1, 0.15, 0.6, 0.8])  # [left, bottom, width, height]
    
    # --- Plot heatmaps ---
    sns.heatmap(
        neg_data.reshape(1, -1),
        cmap='Reds',
        vmin=0, vmax=1,
        annot=False,
        cbar=False,
        ax=ax
    )
    sns.heatmap(
        pos_data.reshape(1, -1),
        cmap='Blues',
        vmin=0, vmax=1,
        annot=False,
        cbar=False,
        ax=ax
    )
    
    # --- Calculate dynamic colorbar positions ---
    plot_right = 0.1 + 0.6  # ax left + ax width
    cbar_width = 0.02
    gap = 0.04
    
    # Position first colorbar
    cax1 = fig.add_axes([
        plot_right + gap,        # left
        0.15,                   # bottom
        cbar_width,              # width
        0.7                     # height
    ])
    
    # Position second colorbar
    cax2 = fig.add_axes([
        plot_right + gap*2 + cbar_width,  # left
        0.15,                             # bottom
        cbar_width,                        # width
        0.7                               # height
    ])
    
    # Create colorbars
    plt.colorbar(
        plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(0, 1)),
        cax=cax1,
        label='Positive Correlation Frequency'
    )
    plt.colorbar(
        plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(0, 1)),
        cax=cax2,
        label='Negative Correlation Frequency'
    )
    
    # --- Axis formatting ---
    max_features = len(pos_data)
    label_step = max(1, max_features // 30)
    xticks = np.arange(0, max_features, label_step)
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(xticks, rotation=45, ha='right', fontsize=9)
    titlee = f'Feature Consensus: {format_title(key)}'
    ax.set_title(titlee, fontsize=12)
    ax.set_xlabel('Feature Index', fontsize=10)
    ax.set_ylabel('')
    if show:
        plt.show(block=True)  # block=True keeps the plot open
    if save:
        plt.savefig(f'{SAVE_DIR}/{titlee}.png')
    return
    
def format_title(key: str, no_corref=False) -> str:
    """Format the title for the plot."""
    if not no_corref:
        title = key.replace('ind', 'index').replace('corrcoef', 'correlation coefficient').replace('_', ' ').title()
    else:
        title = key.replace('_', ' ').replace('ind', 'index').replace(' corrcoef', '').title()
    return title
    


# Train model
# model = XGBRegressor(tree_method="hist", n_estimators=500)
# model.fit(X, y)

# # Native feature importance
# plt.barh(range(X.shape[1]), model.feature_importances_)
# plt.yticks(range(X.shape[1]), feature_names)
# plt.title("XGBoost Feature Importance")
# plt.show()

# # Permutation importance
# result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
# sorted_idx = result.importances_mean.argsort()
# plt.boxplot(result.importances[sorted_idx].T, vert=False)
# plt.yticks(range(X.shape[1]), [feature_names[i] for i in sorted_idx])
# plt.title("Permutation Importance")
# plt.show()

# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)

# # Summary plot
# shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar")

# # Dependence plots for top features
# shap.dependence_plot(
#     "most_important_feature", 
#     shap_values, 
#     X, 
#     feature_names=feature_names,
#     interaction_index=None
# )

def plot_feature_correlation_boxplots(metrics_dict, key, N=10, spacing=1.0, show=True, save=False):
    """Plot box plots for feature correlations across all points, every Nth feature."""
    
    plt.ioff()
    num_features = metrics_dict['params']['num_features']
    points = metrics_dict['params']['valid_points']
    
    # Collect correlation data for each feature across all points
    feature_correlations = [[] for _ in range(num_features)]
    for point in points:
        corr_coeffs = metrics_dict[str(point)][key]
        for feature_idx in range(num_features):
            feature_correlations[feature_idx].append(corr_coeffs[feature_idx])
    
    # Select every Nth feature
    selected_features = list(range(0, num_features, N))
    selected_data = [feature_correlations[idx] for idx in selected_features]
    
    # Adjust positions based on spacing
    positions = [idx * spacing for idx in selected_features]
    
    # Create the plot with dynamic width
    # fig_width = max(12, len(selected_features) * 0.5 * spacing)  # 26 originally
    fig_width = 26
    
    plt.figure(figsize=(fig_width, 10))
    
    boxplot = plt.boxplot(
        selected_data,
        positions=positions,
        widths=4 * spacing,
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color="black", linewidth=2, )
    )
    
    # Customize box colors
    for box in boxplot['boxes']:
        box.set_facecolor('lightblue')
    
    # spacing = spacing * 2  # Adjust spacing for new positions
    # selected_features = list(range(0, num_features, 2*N))
    # positions = [idx * spacing for idx in selected_features]
    
    plt.xlabel('Feature Index', size=30)
    plt.ylabel('Correlation Coefficient', size=30)
    title = f'Box Plots of {format_title(key)} for Every {N}th Feature'
    plt.title(title, size=35, pad=20)  # Add padding to prevent title overlap
    
    # Set x-axis limits and ticks
    plt.xlim(min(positions) - 5*spacing, max(positions) + 5*spacing)
    plt.xticks(positions[::2], selected_features[::2], rotation=45, size=25)
    plt.yticks(size=25)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Fix layout and margins
    plt.tight_layout()  # Automatically adjust subplot margins
    plt.subplots_adjust(bottom=0.2)  # Extra space for x-axis label
    
    # Save before closing
    if save:
        os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure directory exists
        filename = title.replace(" ", "_") + ".png"  # Sanitize filename
        plt.savefig(os.path.join(SAVE_DIR, filename), bbox_inches='tight')  # Prevent cutoff
    
    if show:
        plt.show(block=True)
    plt.close()
    
if __name__ == '__main__':

    metrics_dict = calculate_metrics(full_run)

    # for key in corrcoef_keys:
        # plot_global_correlations(metrics_dict, key, show=True, save=False)
    
    # for key in corrcoef_keys:
    #     plot_correlations(metrics_dict, key, show=True, save=False)
    
    # for key in metrics_dict['consensus'].keys():
    #     plot_consensus_heatmap(metrics_dict, key, show=True, save=False)
    
    # plot_global_correlations(metrics_dict, 'objectness_score_corrcoef', show=False, save=True)
    
    plot_feature_correlation_boxplots(metrics_dict, 
                                      'objectness_score_corrcoef', 
                                      5, 
                                      1.0,
                                      show=True, 
                                      save=True)
    