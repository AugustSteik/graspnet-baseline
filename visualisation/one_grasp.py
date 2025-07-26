from graspnetAPI import GraspNet, Grasp, GraspGroup
import open3d as o3d
import numpy as np


# transform grasp
g = Grasp() # simple Grasp
frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.02, (0.05, -0.05, -0.05))

# Grasp before transformation
o3d.visualization.draw_geometries([g.to_open3d_geometry(), frame])
# g.translation = np.array((0,0,0.01))

# # setup a transformation matrix
# T = np.eye(4)
# T[:3,3] = np.array((0.01, 0.02, 0.03))
# T[:3,:3] = np.array([[0,0,1.0],[1,0,0],[0,1,0]])
# g.transform(T)

# # Grasp after transformation
# o3d.visualization.draw_geometries([g.to_open3d_geometry(), frame])

# g1 = Grasp()
# gg = GraspGroup()
# gg.add(g)
# gg.add(g1)

# # GraspGroup before transformation
# o3d.visualization.draw_geometries([*gg.to_open3d_geometry_list(), frame])

# gg.transform(T)

# # GraspGroup after transformation
# o3d.visualization.draw_geometries([*gg.to_open3d_geometry_list(), frame])