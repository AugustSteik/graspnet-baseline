import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from graspnetAPI import GraspNet, Grasp, GraspGroup
import open3d as o3d
import cv2

gl_path = "dataset\\grasp_label\\000_labels.npz"
tol_path = "dataset\\tolerance\\000_tolerance.npy"

label = np.load(gl_path) # Labels on objects 
tolerance = np.load(tol_path)
                                # 300 - view id, 12 - in-plane rotation, 4 - depth id
grasp_labels = (label['points'].astype(np.float32),  # (3459, 3) coordinates
                label['offsets'].astype(np.float32), # (3459, 300, 12, 4, 3) in-plane rotation, depth and width of the gripper respectively in the last dimension.
                label['scores'].astype(np.float32),  # (3459, 300, 12, 4) minimum coefficient of friction between the gripper and object - lower score is better, but -1 means the pose is unacceptable
                label['collision'],                  # (3459, 300, 12, 4) Bool mask for if the grasp pose collides with the model
                tolerance)  # tolerance for each point 

def plot_grasp_points():
    grasp_points = grasp_labels[0]
    xp, yp, zp = grasp_points[:, 0], grasp_points[:, 1], grasp_points[:, 2]  # where each point is [x, y, z]


    point_idx = 0
    grasp_point = grasp_points[point_idx]  # get the first grasp point
    selected_vectors = grasp_labels[1][point_idx, :, 0, 0, :]  # Shape [300, 3]


    # Prepare data for plotly
    x = [grasp_point[0]] * len(selected_vectors)
    y = [grasp_point[1]] * len(selected_vectors)
    z = [grasp_point[2]] * len(selected_vectors)

    u = selected_vectors[:, 0]  # X-component of vectors
    v = selected_vectors[:, 1]  # Y-component of vectors
    w = selected_vectors[:, 2]  # Z-component of vectors


    # Create a 3D quiver-like plot
    fig = go.Figure()

    # Add grasp point
    fig.add_trace(go.Scatter3d(
        x=[grasp_point[0]], y=[grasp_point[1]], z=[grasp_point[2]],
        mode='markers',
        marker=dict(size=5, color='red'),
        name='Grasp Point'
    ))

    fig.add_trace(go.Scatter3d(x=xp, y=yp, z=zp))

    i = 1
    fig.add_trace(go.Cone(
        x=[x[i]], y=[y[i]], z=[z[i]],
        u=[u[i]], v=[v[i]], w=[w[i]],
        sizemode="scaled", sizeref=0.2,
        anchor="tail", colorscale="Blues",
        opacity=0.5
    ))
            
    # Layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        title="Grasp Point and Approach Vectors",
    )

    fig.show()

if __name__ == '__main__':
    
    # plot_grasp_points()
    
    
    gnet_root = os.path.join(os.path.dirname(__file__), 'dataset')
    sceneId = 1
    annId = 3

    # initialize a GraspNet instance  
    g = GraspNet(gnet_root, camera='kinect', split='train')

    # load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
    _6d_grasp = g.loadGrasp(sceneId = sceneId, annId = annId, format = '6d', camera = 'kinect', fric_coef_thresh = 0.2)
    print('6d grasp:\n{}'.format(_6d_grasp))

    # _6d_grasp is an GraspGroup instance defined in grasp.py
    print('_6d_grasp:\n{}'.format(_6d_grasp))