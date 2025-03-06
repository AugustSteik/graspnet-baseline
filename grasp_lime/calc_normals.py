import open3d as o3d
import numpy as np

def load_point_cloud(file_path):
    """Loads a point cloud from .txt, .npy, or .ply."""
    if file_path.endswith('.txt'):
        points = np.loadtxt(file_path, delimiter=',')  # XYZ format
    elif file_path.endswith('.npy'):
        points = np.load(file_path)
    elif file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    else:
        raise ValueError("Unsupported file format. Use .txt, .npy, or .ply")
    
    return points

def compute_normals(points, k_neighbors=30):
    """Computes normals for a point cloud using Open3D."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    # Estimate normals
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors))
    
    # Orient normals consistently
    pc.orient_normals_consistent_tangent_plane(k_neighbors)
    
    # Extract normals as numpy array
    normals = np.asarray(pc.normals)
    
    return np.hstack((points, normals))  # Concatenate XYZ with NX, NY, NZ

def check_normals():
    ply_path = "/home/as_admin/development/LIME-3D/data/ModelNet40_converted/normals/airplane/airplane_0001.ply"  # Change to your actual file
    pc = o3d.io.read_point_cloud(ply_path)

    # Check if normals exist
    if not pc.has_normals():
        print("⚠️ The PLY file does NOT contain normals!")
    else:
        print("✅ The PLY file contains normals!")

    # Convert to NumPy array
    xyz = np.asarray(pc.points)  # Should have shape (N, 3)
    normals = np.asarray(pc.normals)  # Should have shape (N, 3)

    # Print a few points and normals
    print("First 5 points (XYZ):", xyz[:5])
    print("First 5 normals (NX, NY, NZ):", normals[:5])
    return

def save_point_cloud(points_with_normals, output_path):
    """Saves the point cloud with normals to .txt, .npy, or .ply."""
    if output_path.endswith('.txt'):
        np.savetxt(output_path, points_with_normals, delimiter=',', fmt="%.6f")
    elif output_path.endswith('.npy'):
        np.save(output_path, points_with_normals)
    elif output_path.endswith('.ply'):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(points_with_normals[:, :3])
        pc.normals = o3d.utility.Vector3dVector(points_with_normals[:, 3:])
        o3d.io.write_point_cloud(output_path, pc)
    else:
        raise ValueError("Unsupported file format. Use .txt, .npy, or .ply")

# Example Usage
input_file = "/home/as_admin/development/LIME-3D/data/ModelNet40_converted/airplane/airplane_0001.ply"  # Change to your file
output_file = "/home/as_admin/development/LIME-3D/data/ModelNet40_converted/normals/airplane/airplane_0001.ply"  # Change output format if needed

points = load_point_cloud(input_file)
points_with_normals = compute_normals(points)
save_point_cloud(points_with_normals, output_file)

print(f"✅ Saved point cloud with normals to: {output_file}")
