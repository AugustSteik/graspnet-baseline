import torch
import json
import numpy as np
import os


ROOT = os.path.dirname(os.path.abspath(__file__))

with open(f'{ROOT}/scene_100_backbone_output.json', 'r') as f:
    dummy_data = json.load(f)

for k, v in dummy_data.items():
    dummy_data[k] = torch.tensor(v['data'])
    
    
print(dummy_data.keys())