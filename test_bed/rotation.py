import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def rotate_matrix(batch_data, alpha=0, beta=0, gamma=0):
    
    rotated_batch_data = np.zeros(batch_data.shape, dtype=np.float32)
    alpha = alpha * np.pi / 180
    beta = beta * np.pi / 180
    gamma = gamma * np.pi / 180
    
    rot_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                      [np.sin(gamma), np.cos(gamma), 0],
                      [0, 0, 1]])

    rot_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])
    
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(alpha), -np.sin(alpha)],
                      [0, np.sin(alpha), np.cos(alpha)]])
    
    rotation_matrix = np.matmul(np.matmul(rot_z, rot_y), rot_x)
    
    try:
        _ = batch_data.shape[2]
    except IndexError:
            shape_pc = batch_data
            # shape_pc = shape_pc.reshape((-1, 3))
            rotated_batch_data = np.dot(shape_pc, rotation_matrix)
            # rotated_batch_data[batch, ...] = rotated_data
    else:
        for batch in range(batch_data.shape[0]):
            shape_pc = batch_data[batch, ...]
            # shape_pc = shape_pc.reshape((-1, 3))
            rotated_data = np.dot(shape_pc, rotation_matrix)
            rotated_batch_data[batch, ...] = rotated_data
        
    return rotated_batch_data

import numpy as np

def rotate_matrix_gpt(batch_data, alpha=0, beta=0, gamma=0):
    """
    Rotates a batch of 3D point clouds by specified Euler angles (degrees).
    
    :param batch_data: (B, N, 3) numpy array containing point clouds
    :param alpha: Rotation around x-axis in degrees
    :param beta: Rotation around y-axis in degrees
    :param gamma: Rotation around z-axis in degrees
    :return: Rotated batch data (B, N, 3)
    """
    # Convert degrees to radians
    alpha, beta, gamma = np.radians([alpha, beta, gamma])

    # Rotation matrices
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(alpha), -np.sin(alpha)],
                      [0, np.sin(alpha), np.cos(alpha)]])

    rot_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]])

    rot_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                      [np.sin(gamma), np.cos(gamma), 0],
                      [0, 0, 1]])

    # Correct order of matrix multiplication
    rotation_matrix = rot_z @ rot_y @ rot_x  # Rotation order: X → Y → Z

    # Apply rotation to entire batch
    rotated_batch_data = np.einsum('ij,bnj->bni', rotation_matrix, batch_data)

    return rotated_batch_data


if __name__ == '__main__':
    
    # Making an L-shaped pc
    B, N = 1, 1024
    random_batch = np.zeros((B, N, 3), dtype=np.float32)
    # random_batch[0, :512, 0] = [i for i in range(512)]
    # random_batch[0, 512:, 2] = [i for i in range(512)]
    random_batch[0, :512, 0], random_batch[0, 512:, 2] = np.arange(512), np.arange(512)  # GPT

    rotated_matrix = rotate_matrix_gpt(random_batch, 45, 0, 0)
    
    df_original = pd.DataFrame(np.squeeze(random_batch, axis=0), columns=['x', 'y', 'z'])
    
    fig = go.Figure(data = [go.Scatter3d(x=df_original['x'], y=df_original['y'], z=df_original['z'])])
    
    
    df_rotated = pd.DataFrame(np.squeeze(rotated_matrix), columns=['x', 'y', 'z'])
    
    fig.add_trace(go.Scatter3d(x=df_rotated['x'], y=df_rotated['y'], z=df_rotated['z']))
    
    fig.show()
    
    