from pc_dataset import PCDataset
import open3d as o3d
import plotly.graph_objects as go


object_ids = list(range(35))
PC_DATASET = PCDataset(object_ids=object_ids)

print(PC_DATASET.__len__())
sample = PC_DATASET.__getitem__(28)
print(sample.keys())


fig = go.Figure()


grasp_points_trace = go.Scatter3d(
    x=sample['grasp_points_list'][:, 0],
    y=sample['grasp_points_list'][:, 1],
    z=sample['grasp_points_list'][:, 2],
    hovertext='grasp',
    mode='markers',
    marker=dict(size=5, color='red', opacity=1.0),
    name="Grasping Points"
)

# point cloud trace 
point_cloud_trace = go.Scatter3d(
    x=sample['point_clouds'][:, 0],
    y=sample['point_clouds'][:, 1],
    z=sample['point_clouds'][:, 2],
    mode='markers',
    marker=dict(size=3, color='blue', opacity=0.8),
    name="Object Point Cloud"
)

# object_normal_traces = []
# arrow_scale = 0.01
# for i in range(len(sample['point_clouds'])):
#     start = sample['point_clouds'][i]
#     end = start + sample['point_cloud_normals'][i] * arrow_scale
#     object_normal_traces.append(
#         go.Scatter3d(
#             x=[start[0], end[0]],
#             y=[start[1], end[1]],
#             z=[start[2], end[2]],
#             mode='lines',
#             line=dict(color='green', width=2),
#             name=f"Point Cloud Normals" if i == 0 else None,  # Name only the first arrow
#             showlegend=(i == 0)  # Only show legend for the first arrow
#         )
#     )

grasp_normal_traces = []
arrow_scale = 0.01
for i in range(len(sample['grasp_points_list'])):
    start = sample['grasp_points_list'][i]
    end = start + sample['grasp_cloud_normals'][i] * arrow_scale
    grasp_normal_traces.append(
        go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color='pink', width=2),
            name=f"Grasp point normals" if i == 0 else None,  # Name only the first arrow
            showlegend=(i == 0)  # Only show legend for the first arrow
        )
    )
    
fig = go.Figure(data=[point_cloud_trace, grasp_points_trace] + grasp_normal_traces)

fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'  # Keep aspect ratio correct
    ),
    title="3D Point Cloud with Normals and Grasping Points"
)

fig.show()
