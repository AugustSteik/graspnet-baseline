import sys, os
import torch
import copy
import pandas as pd
import numpy as np


GNET_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(GNET_ROOT)
ROOT = os.path.dirname(__file__)
sys.path.append(ROOT)

from models.loss import compute_objectness_loss, compute_view_loss, compute_grasp_loss  # grasp loss is multiplied by 0.2 for the overall loss sum

from xai_inference_utils import load_data, load_model, inference_one_batch, check_uniqueness, add_labels
from perturb_features import perturb_features, calculate_feature_metrics


""" 
PLAN:
* 1 feature at a time set to 0, run inference, save 
* 1 feature at a time set to 1.0, run inference, save output  (not just outupt but also the loss)

this means, change 1 and run inference, reset the fp2 features in end points, repeat
"""

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_data(data, column_names, out_format, filename):
    # Save the data to a CSV file
    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=",".join(column_names),  # Join names with commas
        comments="",  # Prevents '#' (default comment char) in header
        fmt=out_format    # Control float precision
    )
    
    
if __name__ == '__main__':

    # Load the model and data
    data_path = f'{ROOT}/scene_100-end_points.json'  # pointnetbackbone-end_points_full.json
    
    approachnet = load_model()
    end_points = load_data(data_path)

    # test set-up
    # NOTE: get: grasp_base_top_view_score from base case - this is the top score per point at the original bast view ind
    test_column_names = ['point_idx', 'feature_idx', 'value', 'objectness_score', 'grasp_top_view_ind', 'grasp_top_view_score', 'grasp_base_top_view_ind_score']
    values = [0.0, 2.0, 4.0, 6.0, 8.0]  # 0, 2, 4, 6, 8
    num_features = 256
    num_points = 1024
    
    # logging set-up
    run_name = 'run_5'
    scene_id = '100'
    SAVE_DIR = os.path.join(ROOT, scene_id, run_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # getting base case values
    
    base_case_column_names = ['point_idx', 'objectness_label'] + [str(i).zfill(3) for i in range(num_features)] + ['objectness_score', 'grasp_top_view_inds', 'grasp_top_view_score']
    base_outputs = inference_one_batch(approachnet, end_points)
    
    base_case_matrix = np.zeros((num_points, len(base_case_column_names)))
    
    # objectness_score = sigmoid(base_outputs['objectness_score'].detach().cpu().squeeze()[1, :].numpy())  # class at index 1 is the positive class in binary classification - OLD
    
    # objectness_logits = base_outputs['objectness_score'].detach().cpu()
    # objectness_score = torch.softmax(objectness_logits, dim=1)[:, 1, :].squeeze().numpy()
    
    objectness_score = base_outputs['objectness_score'].detach().cpu()[:, 1, :].squeeze().numpy()  # getting logits for graspable class
    
    top_view_score = base_outputs['grasp_top_view_score'].detach().cpu().squeeze().numpy()
    
    top_view_inds = base_outputs['grasp_top_view_inds'].detach().cpu().squeeze().numpy().astype(int)  # (1024, )
    fp2_inds = base_outputs['fp2_inds'].detach().cpu().squeeze().numpy()
    objectness_label = base_outputs['objectness_label'].detach().cpu().squeeze().numpy()[fp2_inds]
    
    base_case_matrix[0: num_points, 0] = np.array(range(num_points), dtype=int)  # seems to work
    base_case_matrix[0: num_points, 1] = objectness_label
    base_case_matrix[0: num_points, 2:num_features+2] = base_outputs['fp2_features'].detach().cpu().squeeze().numpy().T  # works 
    base_case_matrix[0: num_points, num_features+2] = objectness_score
    base_case_matrix[0: num_points, num_features+3] = top_view_inds
    base_case_matrix[0: num_points, num_features+4] = top_view_score # all checks out
    
    out_format = '%d,%d,' + ''.join(['%.17f,' for _ in range(num_features)]) + '%.17f,%d,%.17f'
    save_data(base_case_matrix, base_case_column_names, out_format, os.path.join(SAVE_DIR, 'base_case.csv'))
    # can be loaded using: np.loadtxt(os.path.join(SAVE_DIR, 'base_case.csv'), delimiter=',', skiprows=1)
    
    base_case_dict = {'top_view_ind_per_point': top_view_inds,
                      'objectness_label': objectness_label,
                      }
    
    # setting-up logging matrix
    num_rows = num_features * num_points * len(values)
    log_matrix = np.zeros((num_rows, len(test_column_names)))
    
    point_idxs = np.array(range(num_points), dtype=int)
    current_row = 0
    for value in values:
        for feature_idx in range(num_features):
            # save original value of the feature
            original_value = end_points['fp2_features'].detach().clone()[:, feature_idx, :]
            # perturb the feature
            end_points['fp2_features'][:, feature_idx, :] = value
            
            outputs = inference_one_batch(approachnet, end_points)
            
            end_row = current_row + num_points

            # objectness_score = sigmoid(outputs['objectness_score'].detach().cpu().squeeze()[1, :].numpy()) OLD
            objectness_score = outputs['objectness_score'].detach().cpu()[:, 1, :].squeeze().numpy()
            top_view_score = outputs['grasp_top_view_score'].detach().cpu().squeeze().numpy()
            grasp_base_top_view_ind_score = outputs['view_score'].detach().cpu().squeeze().numpy()[np.arange(num_points), base_case_dict['top_view_ind_per_point']]  # getting the score of the view that was the best in the base 

            log_matrix[current_row: end_row, 0] = point_idxs
            log_matrix[current_row: end_row, 1] = feature_idx
            log_matrix[current_row: end_row, 2] = value
            log_matrix[current_row: end_row, 3] = objectness_score
            log_matrix[current_row: end_row, 4] = outputs['grasp_top_view_inds'].detach().cpu().squeeze().numpy()  # top view ind
            log_matrix[current_row: end_row, 5] = top_view_score
            log_matrix[current_row: end_row, 6] = grasp_base_top_view_ind_score  # grasp base top view ind score
            
            end_points['fp2_features'].detach().clone()[:, feature_idx, :] = original_value  # reset the perturbed feature
            
            current_row = end_row
            # test_format = '%d,%d,%.17f,%.17f,%d,%.17f,%.17f'
            # save_data(log_matrix, test_column_names, test_format, os.path.join(SAVE_DIR, 'full_run.csv'))
        print(f'Finished value {value}')
    test_format = '%d,%d,%.17f,%.17f,%d,%.17f,%.17f'
    save_data(log_matrix, test_column_names, test_format, os.path.join(SAVE_DIR, 'full_run.csv'))
    
    # feature_stats = calculate_feature_metrics(end_points)  # all values fo between 0 and 8, nothing remarkable seen
    