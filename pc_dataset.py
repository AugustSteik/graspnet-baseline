import open3d as o3d
import os
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import numpy as np
import torch
from typing import Any, Tuple, List


class PCDataset(Dataset):
    """Dataset generator for model pointclouds and labels only."""

    def __init__(self, object_ids=[0], num_points=5000, 
                 dataset_root=os.path.join(os.path.dirname(__file__), 'dataset')):
        
        self.object_ids = ['{}'.format(str(obj_id).zfill(3)) for obj_id in object_ids]
        self.num_points = num_points
        self.dataset_root = dataset_root
        o3pointclouds = self._load_pc()
        self.grasp_labels, self.tolerance_labels = self._load_grasp_labels()
        # self.pc_normals = self._estimate_normals(o3pointclouds)
        # self.grasp_normals = self._estimate_normals(self.grasp_labels)
        self.pointclouds = self._convert_to_np(o3pointclouds)
        
        self._downsample([self.pointclouds])
        
    def __len__(self):
        return len(self.grasp_labels)
    
    def __getitem__(self, index):
        return self.get_data_label(index)
    
    def _load_grasp_labels(self):
        """Loads a list of model grasp and tolerance labels as np arrays."""
        grasp_labels = []
        tolerance_labels = []
        # Load model grasp labels - points, offsets, collision, scores
        for x in tqdm(self.object_ids, desc='Loading model grasp labels...'):
            object_label_path = os.path.join(self.dataset_root,'grasp_label', f'{str(x)}_labels.npz')
            grasp_labels.append(np.load(object_label_path))
            
        for x in tqdm(self.object_ids, desc='Loading grasp tolerance labels...'):
            object_tol_path = os.path.join(self.dataset_root, 'tolerance', f'{str(x)}_tolerance.npy')
            tolerance_labels.append(np.load(object_tol_path))
        return grasp_labels, tolerance_labels
        
    def _load_pc(self) -> 'list[o3d.geometry.PointCloud]':
        """Returns a list of model o3d pointclouds."""
        clouds = []
        for x in tqdm(self.object_ids, desc='Loading model pointclouds...'):
            object_pc_path = os.path.join(self.dataset_root,'models', str(x), 'nontextured.ply')
            o3cloud = o3d.io.read_point_cloud(object_pc_path)
            clouds.append(o3cloud)
        return clouds
    
    def _estimate_normals(self, clouds: 'list') -> 'list[np.ndarray]':
        """Estimate normal vectors for object point cloud and grasping points."""
        out_clouds = []
        # Compute normals for object point cloud
        if isinstance(clouds[0], o3d.geometry.PointCloud):
            for cloud in clouds:
                cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                out_clouds.append(np.asarray(cloud.normals))
            return out_clouds
        else:
            # Compute normals for grasp points
            for cloud in clouds:
                centroid = np.mean(cloud['points'], axis=0)
                temp_cloud = o3d.geometry.PointCloud()
                temp_cloud.points = o3d.utility.Vector3dVector(cloud['points'])
                temp_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                temp_cloud.orient_normals_towards_camera_location(camera_location=centroid)
                out_clouds.append(np.asarray(temp_cloud.normals))
            return out_clouds
    
    def _convert_to_np(self, clouds: o3d.geometry.PointCloud) -> 'list[Any]':
        out_clouds = []
        for cloud in clouds:
            out_clouds.append(np.asarray(cloud.points))
        return out_clouds
    
    def _downsample(self, clouds_and_normals: 'list[list[np.ndarray]]') -> None:
        """Downsample point cloud and corresponding normals."""
        if len(clouds_and_normals) == 2:
            clouds = clouds_and_normals[0]
            normals = clouds_and_normals[1]
            for idx, cloud in enumerate(clouds):
                if len(cloud) > self.num_points:
                    idxs = np.random.choice(len(cloud), self.num_points, replace=False)
                    cloud = cloud.copy()
                    normals[idx] = normals[idx].copy()
                    clouds[idx] = cloud[idxs]
                    normals[idx] = normals[idx][idxs]
            return
        else:
            clouds = clouds_and_normals[0]
            for idx, cloud in enumerate(clouds):
                if len(cloud) > self.num_points:
                    idxs = np.random.choice(len(cloud), self.num_points, replace=False)
                    cloud = cloud.copy()
                    clouds[idx] = cloud[idxs]
            return
        
    def _get_best_scores(self, grasp_scores: 'Any') -> 'Tuple[List[float], List[Tuple[int, ...]]]':

        best_point_score = np.min(grasp_scores[grasp_scores > 0])
        best_score_idxs = np.array(np.where(grasp_scores == best_point_score)).T
        best_score_values = grasp_scores[grasp_scores == best_point_score]

        return best_score_values, best_score_idxs
    
    def _generate_objectness_labels(self, length):
        return np.ones(length, dtype=np.int64)
            
    def get_data_label(self, index: int) -> dict:
        ret_dict = {}
        grasp_points_list = []
        grasp_offsets_list = []
        grasp_labels_list = []
        grasp_tolerance_list = []
        grasp_labels_list = []
        best_scores_list = []
        best_scores_idxs_list = []
        
        label = self.grasp_labels[index]
        
        grasp_tolerance_list.append(self.tolerance_labels[index])
        grasp_points_list.append(label['points'])
        grasp_offsets_list.append(label['offsets'])
        grasp_labels_list.append(label['scores'])
        
        best_scores, best_score_idxs = self._get_best_scores(label['scores'])
        best_scores_list.append(best_scores)
        best_scores_idxs_list.append(best_score_idxs)
        
        ret_dict['point_clouds'] = self.pointclouds[index].astype(np.float32)
        ret_dict['objectness_label'] = self._generate_objectness_labels(len(self.pointclouds[index]))
        ret_dict['grasp_tolerance_list'] = grasp_tolerance_list
        ret_dict['grasp_points_list'] = grasp_points_list
        ret_dict['grasp_offsets_list'] = grasp_offsets_list
        ret_dict['grasp_labels_list'] = grasp_labels_list
        ret_dict['best_scores_list'] = best_scores_list
        ret_dict['best_scores_idxs_list'] = best_scores_idxs_list
        
        # TODO: Look into looking the normals
        # ret_dict['point_cloud_normals'] = self.pc_normals[index].astype(np.float32)
        # ret_dict['grasp_cloud_normals'] = self.grasp_normals[index].astype(np.float32)

        return ret_dict
