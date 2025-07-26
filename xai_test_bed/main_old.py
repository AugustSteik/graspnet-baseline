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
* 1 feature at a time set to 0, run inference, save output
* 1 feature at a time set to 1.0, run inference, save output  (not just outupt but also the loss)

this means, change 1 and run inference, reset the fp2 features in end points, repeat
"""
def log_results(columns, end_points):
    features = end_points['fp2_features'].detach().squeeze().transpose(0, 1).cpu().tolist()  # was [1, 256, 1024]
    
    objectness_score = end_points['objectness_score'].detach().cpu().squeeze()[0, :].numpy()  # was [1, 2, 1024]
    objectness_score = (1 / (1 + np.exp(objectness_score))).tolist()  # sigmoid
    view_scores = end_points['view_score'].detach().cpu().squeeze().tolist()  # was [1, 1024, 300]
    
    objectness_loss = end_points['loss/stage1_objectness_loss'].detach().cpu().squeeze().tolist()  # was [1, 1024]
    
    view_loss = end_points['loss/stage1_view_loss'].detach().cpu().squeeze().tolist()  # was [1, 1024]

    # results_df.loc[len(results_df)] = features + objectness_loss + view_loss + objectness_score + view_scores
    rows = []
    for idx in range(len(features)):
        row = features[idx] + [objectness_loss[idx], view_loss[idx], objectness_score[idx]] + view_scores[idx]
        rows.append(row)
    return rows
    
if __name__ == '__main__':

    approachnet = load_model()
    # end_points = load_data(f'{ROOT}/scene_100-end_points.json')
    end_points = load_data(f'{GNET_ROOT}/variable_logs/pointnetbackbone-end_points_full.json')
    
    end_points_original = copy.deepcopy(end_points)
    fp2_feature_original = end_points['fp2_features'].detach().clone()
    
    all_rows = []
    
    min = 0
    max = 8
    step = 2
    
    base_outputs = inference_one_batch(approachnet, end_points)
    for k, v in base_outputs.items():
        if isinstance(v, torch.Tensor):
            print(f'{k}: {v.size()}')
    o_loss, base_outputs = compute_objectness_loss(base_outputs, reduction='none')
    v_loss, base_outputs = compute_view_loss(base_outputs, pointwise=True)
    # g_loss, base_outputs = compute_grasp_loss(base_outputs)
    COLUMNS = ['feature_' + str(i) for i in range(256)] + ['objectness_loss', 'view_loss', 'objectness_score'] + ['view_' + str(i) for i in range(300)]
    # results_df = pd.DataFrame(columns=columns)  # 0:255-features, 256:258-point-wise metrics, 259:559-view-wise metrics
    
    rows = log_results(COLUMNS, base_outputs)
    results_df = pd.DataFrame(rows, columns=COLUMNS)
    results_df.to_csv(os.path.join(SAVE_DIR, 'base_000.csv'), index=False)
    del results_df
    del rows
    
    ### used for runs 1-3:
    # for i in range(end_points['fp2_features'].shape[1]):  # 0 to 256
    #     for j in range(min, max+1, step):  # 0 to 8
    #         # deviation = j - fp2_feature_original[:, :, i]
    #         end_points['fp2_features'] = end_points_original['fp2_features'].clone() # reset the features
    #         end_points['fp2_features'][:, i, :] = j  # change the i-th feature to j, for all points
            
    #         outputs = inference_one_batch(approachnet, end_points)
            
    #         o_loss, outputs = compute_objectness_loss(outputs, reduction='none')
    #         v_loss, outputs = compute_view_loss(outputs, pointwise=True)
    #         # g_loss, outputs = compute_grasp_loss(outputs)
            
    #         all_rows = all_rows + log_results(COLUMNS, outputs)
        
    #     if i % 10 == 0:  # edge case for 0 LOL
    #         print(f'Processed {i} features')
    #         results_df = pd.DataFrame(all_rows, columns=COLUMNS)
    #         filename = f'{i}'.zfill(3)
    #         results_df.to_csv(os.path.join(SAVE_DIR, f'perturbation_{filename}.csv'), index=False)
    #         all_rows = []
    #         del results_df
    ### end of used for runs 1-3
    results_df = pd.DataFrame(all_rows, columns=COLUMNS)
    filename = f"{end_points['fp2_features'].shape[1]}"
    results_df.to_csv(os.path.join(SAVE_DIR, f'perturbation_{filename}.csv'), index=False)
    del all_rows
    del results_df
    
    ### New approach for run 4:
    # logging set-up
    run_name = 'run_4'
    scene_id = '100'
    SAVE_DIR = os.path.join(ROOT, scene_id, run_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # test set-up
    # NOTE: get: grasp_top_view_inds from base case
    column_names = ['point_idx', 'feature_idx', 'value', 'objectness_score', 'grasp_top_view_ind', 'grasp_top_view_score', 'grasp_base_top_view_score']
    values = [0.0, 2.0, 4.0, 6.0, 8.0]  # 0, 2, 4, 6, 8
    num_features = 256
    num_points = 1024
    
    # setting-up logging matrix
    num_rows = num_features * num_points * len(values)
    num_columns = len(column_names)
    log_matrix = np.zeros((num_rows, num_columns))
    
    # saving data
    np.savetxt(
    'outputs.csv',
    log_matrix,
    delimiter=",",
    header=",".join(column_names),  # Join names with commas
    comments="",  # Prevents '#' (default comment char) in header
    fmt='%d,%d,%.17g,%.17g'    # Control float precision (or use `%d` for integers)
)
    

    
    # feature_stats = calculate_feature_metrics(end_points)  # all values fo between 0 and 8, nothing remarkable seen
    