from models.graspnet import GraspNet, pred_decode_per_object_view_logits
import torch
import numpy as np

from record_something import log_variable, varname

class WrappedGraspNet(GraspNet):
    """ Graspnet dupe """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.idx = 0
        self.best_point = None
        
    def forward(self, end_points, object_id=None):
        if object_id is not None:
            self.object_id = object_id
        if isinstance(end_points, dict) :
            self.end_points = super().forward(end_points)
            # sorted_fp2_xyz, sorted_point_idxs = self.sort_by_xyz(self.end_points['fp2_xyz'].squeeze(0))
            # self.end_points = self.reorder_end_points(self.end_points, sorted_point_idxs)
            return self.end_points
        else:
            self.end_points['point_clouds'] = end_points  # end_points here is an input piontcloud
            
            self.end_points = super().forward(self.end_points)
            # sorted_fp2_xyz, sorted_point_idxs = self.sort_by_xyz(self.end_points['fp2_xyz'].squeeze(0))
            # self.end_points = self.reorder_end_points(self.end_points, sorted_point_idxs)
            
            grasps_per_object, best_grasp_per_object, view_logits_per_object, best_grasp_view_logits_per_object, point_idxs_per_obj = pred_decode_per_object_view_logits(self.end_points)
            
            # end_points.update({'best_grasp_per_object': best_grasp_per_object,
            #                   'view_logits_per_object': view_logits_per_object,
            #                   'best_grasp_view_logits_per_object': best_grasp_view_logits_per_object})
            # view_logits = best_grasp_view_logits_per_object[self.object_id]
            
            # sorted_view_scores = end_points['view_score'][:, sorted_point_idxs, :]
            flattened_view_logits = torch.flatten(self.end_points['view_score']).unsqueeze(0)
            if self.best_point is None:
                point_idx = point_idxs_per_obj[self.object_id][self.find_best_grasp_idx(grasps_per_object, self.object_id)][0]
                view_idx = self.end_points['grasp_top_view_inds'].squeeze()[point_idx]
                self.best_point = [point_idx, view_idx.item()]
                # grasp_facts = {'best_point': }
            return flattened_view_logits, self.best_point
        # NOTE: the view_logits is good here, and needed. the actual predicted view isnt used in lime. NOW, outpuutting index of the best initial point on a given object
    
    def find_best_grasp_idx(self, gpo, object_id):
        score_list = np.array([item[0].item() for item in gpo[object_id]])
        score_idx = np.argmax(score_list)
        return score_idx
    
    def sort_by_xyz(self, pc):
        """
        Sorts point cloud by (x, y, z), ensuring points are ordered in a structured way.

        Arguments:
            pc: NumPy array or torch.Tensor of shape (N, 3).

        Returns:
            sorted_pc: NumPy array of shape (N, 3) sorted by (x, y, z).
            idx_order: The permutation of indices that leads to sorted_pc.
        """
        if isinstance(pc, torch.Tensor):
            pc = pc.cpu().numpy()
        idx_order = np.lexsort((pc[:,2], pc[:, 1], pc[:, 0]))  # Sort by (x, then y, then z)
        # idx_order = np.lexsort((pc[:,0], pc[:, 1], pc[:, 2]))  # Sort by (z, then y, then x)
        sorted_pc = pc[idx_order]
        return sorted_pc, idx_order
    
    def reorder_end_points(self, end_points, idx_order):
        for k, v in end_points.items():
            if not isinstance(v, torch.Tensor):
                continue
            if v.shape[1] != 1024:
                continue
            else:
                try:
                    end_points[k] = v[:, idx_order, :]
                except Exception:
                    end_points[k] = v[:, idx_order]
        return end_points

# class WrappedGraspNet2(GraspNet):
#     """ Graspnet dupe """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#     def forward(self, end_points, object_id=None):
        