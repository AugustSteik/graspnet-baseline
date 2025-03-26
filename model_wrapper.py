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



from models.graspnet import GraspNet, pred_decode_per_object_view_logits
import torch
import numpy as np

class WrappedGraspNet2(GraspNet):
    """ Wrapper around GraspNet to freeze or reuse point ordering. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_point = None
        # Cache for sorting index so we don't re-compute each time
        self.cached_idx_order = None

    def forward(self, end_points_or_pc, object_id=None, force_resort=False, skip_decode=False):
        """
        forward() can accept either:
          1) A dictionary 'end_points_or_pc' (typical usage in some code)
          2) A raw point cloud Tensor (B, N, 3 or 6) if not a dict.

        Args:
            object_id: If provided, used for picking best grasp points.
            force_resort: If True, re-compute the sorted indices instead of reusing cached_idx_order.
            skip_decode: If True, skip pred_decode_per_object_view_logits() for faster runs.
        """
        # Track the object_id inside the wrapper
        if object_id is not None:
            self.object_id = object_id

        # 1) If input is a dict, we assume it is already an end_points structure
        if isinstance(end_points_or_pc, dict):
            end_points = end_points_or_pc
            # Run the core GraspNet forward
            end_points = super().forward(end_points)

            # Possibly re-sort by xyz if not cached or forced
            if self.cached_idx_order is None or force_resort:
                sorted_fp2_xyz, sorted_point_idxs = self.sort_by_xyz(end_points['fp2_xyz'].squeeze(0))
                self.cached_idx_order = torch.from_numpy(sorted_point_idxs).long().to(end_points['fp2_xyz'].device)
            # Reuse the cached_idx_order (no repeated call to sort_by_xyz)
            end_points = self.reorder_end_points(end_points, self.cached_idx_order)

            # Return the updated end_points
            return end_points

        else:
            # 2) Otherwise, we interpret 'end_points_or_pc' as an input pointcloud
            #    We'll pack it into end_points['point_clouds'] for super().forward
            pc = end_points_or_pc
            end_points = {}
            end_points['point_clouds'] = pc

            # Run the core GraspNet forward
            end_points = super().forward(end_points)

            # Sort only if needed
            if self.cached_idx_order is None or force_resort:
                sorted_fp2_xyz, sorted_point_idxs = self.sort_by_xyz(end_points['fp2_xyz'].squeeze(0))
                self.cached_idx_order = torch.from_numpy(sorted_point_idxs).long().to(end_points['fp2_xyz'].device)
            end_points = self.reorder_end_points(end_points, self.cached_idx_order)

            # Optionally skip the decode logic for efficiency
            if skip_decode:
                # Just flatten the view_score for LIME or other usage
                flattened_view_logits = torch.flatten(end_points['view_score'], start_dim=1)  # shape (B, 1024*num_views)
                return flattened_view_logits, None
            else:
                # Run normal decode
                grasps_per_obj, best_grasp_per_obj, view_logits_per_obj, best_grasp_view_logits_per_obj, point_idxs_per_obj \
                    = pred_decode_per_object_view_logits(end_points)

                flattened_view_logits = torch.flatten(end_points['view_score']).unsqueeze(0)

                # If we haven't stored the best point yet, pick it once
                if self.best_point is None:
                    best_idx = self.find_best_grasp_idx(grasps_per_obj, self.object_id)
                    point_idx = point_idxs_per_obj[self.object_id][best_idx][0]
                    view_idx = end_points['grasp_top_view_inds'].squeeze()[point_idx]
                    self.best_point = [point_idx, view_idx.item()]

                return flattened_view_logits, self.best_point

    def find_best_grasp_idx(self, gpo, object_id):
        score_list = np.array([item[0].item() for item in gpo[object_id]])
        score_idx = np.argmax(score_list)
        return score_idx

    def sort_by_xyz(self, pc):
        """
        Sorts point cloud by (x, y, z).

        Args:
            pc: (N, 3) as a NumPy array or torch.Tensor
        Returns:
            sorted_pc, idx_order
        """
        if isinstance(pc, torch.Tensor):
            pc = pc.detach().cpu().numpy()
        idx_order = np.lexsort((pc[:,2], pc[:,1], pc[:,0]))  # (x, then y, then z)
        sorted_pc = pc[idx_order]
        return sorted_pc, idx_order

    def reorder_end_points(self, end_points, idx_order_torch):
        """
        Reorders all relevant (B, N, ...) tensors in end_points according to idx_order_torch.

        Only reorders keys where the second dimension == 1024 (the seed points).
        """
        for k, v in end_points.items():
            if not isinstance(v, torch.Tensor):
                continue
            # v.shape might be [B, N, ...]
            if v.shape[1] == 1024:
                # Reorder along dimension 1
                try:
                    end_points[k] = v[:, idx_order_torch, ...]
                except Exception:
                    # In case shapes mismatch, do something else or skip
                    pass
        return end_points
