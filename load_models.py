import open3d as o3d

model_path = 'dataset\\models\\000\\textured.obj'

mesh = o3d.io.read_triangle_mesh(model_path)

# Visualize the mesh
o3d.visualization.draw_geometries([mesh])

