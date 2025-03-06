import torch 
import sys

sys.path.append("/home/as_admin/development/graspnet-baseline/models")
sys.path.append("/home/as_admin/development/graspnet-baseline/dataset")

# from backbone import Pointnet2Backbone
from graspnet import GraspNet

from torch.utils.tensorboard.writer import SummaryWriter
from typing import Dict

CHECKPOINT_PATH = "/home/as_admin/development/graspnet-baseline/logs/original/checkpoint-kn.tar"


class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        output_dict = self.model(*args, **kwargs)  # Get the model's raw output

        # 🔹 If the model returns a tuple, extract the first element
        if isinstance(output_dict, tuple):
            output_dict = output_dict[0]  # Extract the dictionary

        # 🔹 Ensure all values in the dictionary are Tensors (replace None)
        output_dict = {k: (v if isinstance(v, torch.Tensor) else torch.zeros(1)) for k, v in output_dict.items()}

        return output_dict  # Must be a dict[str, torch.Tensor]

net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                     cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
checkpoint = torch.load(CHECKPOINT_PATH)
net.load_state_dict(checkpoint['model_state_dict'])
net.to('cuda')
net.eval()
net = ModelWrapper(net)

end_points = {'point_clouds': torch.randn(1, 20000, 3, device='cuda'),
              }


#  tensorboard start
# writer = SummaryWriter("runs/model_architecture")

# writer.add_graph(net, end_points, use_strict_trace=False)
# writer.close()
#  tensorboard end

# class ModelWrapper(torch.nn.Module):
#     def __init__(self, model):
#         super(ModelWrapper, self).__init__()
#         self.model = model

#     def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
#         output_dict = self.model(*args, **kwargs)  # Get the model's raw output

#         # 🔹 If the model returns a tuple, extract the first element
#         if isinstance(output_dict, tuple):
#             output_dict = output_dict[0]  # Extract the dictionary

#         # 🔹 Ensure all values in the dictionary are Tensors (replace None)
#         output_dict = {k: (v if isinstance(v, torch.Tensor) else torch.zeros(1)) for k, v in output_dict.items()}

#         return output_dict  # Must be a dict[str, torch.Tensor]


# outs1 = net(end_points)
# outs2 = wrapped_model(end_points)

traced_model = torch.jit.trace(net, end_points, strict=False)
# print(traced_model.graph)
# scripted_model = torch.jit.script(net)  # Converts model to TorchScript
torch.jit.save(traced_model, "converted_checkpoint_original-kn.onnx")  # Saves architecture + weights
