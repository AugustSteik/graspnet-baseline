import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


ROOT = os.path.dirname(__file__)


def combine_data(run_name):
    DATA_PATH = os.path.join(ROOT, run_name)
    filenames = os.listdir(DATA_PATH)
    filenames.sort(key=lambda x: int(x.split('_')[1][:3]))
    
    df = pd.read_csv(os.path.join(DATA_PATH, filenames[0]))
    
    for filename in filenames[1:]:
        data = pd.read_csv(os.path.join(DATA_PATH, filename))  
        df = pd.concat([df, data], ignore_index=True, copy=False)
        print(filename)
        
    df.to_csv(os.path.join(DATA_PATH, 'combined_results.csv'), index=False)

def get_file_list(path, run_name):
    data_path = os.path.join(path, run_name)
    filenames = os.listdir(data_path)
    # for filname in filenames:
    #     if filename.endswith('.csv'):
    #         filenames.remove(filename)
    filenames.sort(key=lambda x: x.split('_')[1][:3])
    
    for idx, filename in enumerate(filenames):
        filenames[idx] = os.path.join(data_path, filename)
    print(filenames)
    return filenames

if __name__ == '__main__':
    # df = pd.read_csv(os.path.join(ROOT, 'run_3', 'perturbation_000.csv'))
    
    files = get_file_list(ROOT, 'run_3')
    
    ''' Recording format (awkward) '''
    batch_size = 1024
    num_levels = 5  # 0, 2, 4, 6, 8
    num_features_per_run = 10  # given in the code by i % 10
    length_of_tests = num_levels * num_features_per_run
    
    base_df = pd.read_csv(files[0])
    base_case = {idx: [base_df.iloc[idx]] for idx in range(0, batch_size)}  # dict for base case as lists of np arrays
    del base_df
    
    pointwise_columns = {idx: [] for idx in range(0, batch_size)}  # dict for pointwise columns as lists of np arrays
    # pointwise_columns = {10: []}  # only for point 10 (the 11th point)
    
    for filename in files[1:3]:  # last one is combined_results.csv
        print(f"Processing {filename}")
        df = pd.read_csv(filename)
        for test_num in range(0, length_of_tests):
            for key in pointwise_columns.keys():
                try:
                    # difference = np.array(df.iloc[key+(test_num*batch_size)]) - base_case[key][0]
                    pointwise_columns[key].append(np.array(df.iloc[key+(test_num*batch_size)]))
                except:
                    break
        del df
        print(f"Done processing {filename}")
        
    # feature_columns are 0:256
    objectness_loss, view_loss, objectness_score = 256, 257, 258


    while True:
        try:
            point_idx = int(input("Enter the point index (0-1023) to analyse, or -1 to exit, or -2 for all: "))
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
        if point_idx == -1:
            break
        if point_idx == -2:
            all_data = []
            for value in pointwise_columns.values():
                all_data.extend(value)
            data_matrix = np.vstack(all_data)
        elif (point_idx < 0 or point_idx >= batch_size):
            print(f"Invalid point index. Please enter a number between 0 and {batch_size-1}.")
            continue
        else:
            data_matrix = np.vstack(pointwise_columns[point_idx])

        features = data_matrix[:, :256]  # All feature columns
        objectness_scores = np.expand_dims(data_matrix[:, objectness_score], axis=0).T  # Just the objectness_score column
        view_losses = np.expand_dims(data_matrix[:, view_loss], axis=0).T
        
        features_and_objectness_scores = np.hstack((features, objectness_scores)).T  # transpose so rows are features, columns observations
        features_and_view_losses = np.hstack((features, view_losses)).T
        
        cov_matrix_objectness_scores = np.cov(features_and_objectness_scores, rowvar=False)
        corr_matrix_objectness_scores = np.corrcoef(features_and_objectness_scores, rowvar=False)
        
        cov_matrix_view_losses = np.cov(features_and_view_losses, rowvar=False)
        corr_matrix_view_losses = np.corrcoef(features_and_view_losses, rowvar=False)
        
        # Extract just the covariances between features and objectness_score
        # The last row/column contains the covariances with objectness_score
        feature_objectness_cov = cov_matrix_objectness_scores[:-1, -1]
        feature_objectness_corr = corr_matrix_objectness_scores[:-1, -1]
        
        feature_view_cov = cov_matrix_view_losses[:-1, -1]
        feature_view_corr = corr_matrix_view_losses[:-1, -1]
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(256), feature_objectness_cov)
        plt.title(f"Covariance between Features and Objectness Score for Point {'All Points' if point_idx == -2 else point_idx}")
        plt.xlabel("Feature Index")
        plt.ylabel("Covariance")
        plt.show()
        
        plt.figure(figsize=(15, 6))
        plt.bar(range(256), feature_objectness_corr)
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Correlation Between Features and Objectness Score")
        plt.xlabel("Feature Index")
        plt.ylabel("Pearson Correlation Coefficient")
        plt.ylim(-1, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(256), feature_view_cov)
        plt.title(f"Covariance between Features and View Loss for Point {'All Points' if point_idx == -2 else point_idx}")
        plt.xlabel("Feature Index")
        plt.ylabel("Covariance")
        plt.show()
        
        plt.figure(figsize=(15, 6))
        plt.bar(range(256), feature_view_corr)
        plt.axhline(0, color='black', linestyle='--')
        plt.title("Correlation Between Features and View Losses")
        plt.xlabel("Feature Index")
        plt.ylabel("Pearson Correlation Coefficient")
        plt.ylim(-1, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        