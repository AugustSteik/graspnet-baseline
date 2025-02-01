import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import time 
import os

from graspnetAPI import GraspGroup
from dataset.graspnet_dataset import collate_fn
from models.graspnet import pred_decode
from utils.collision_detector import ModelFreeCollisionDetector

from pc_dataset import PCDataset
from models.graspnet import MyGraspNet

CHECKPOINT_PATH = 'C:\\Development\\GraspNet\\graspnet-baseline\\logs\\my_logs\\train_3_mlp_approachnet_full\\checkpoint.tar'
OUTPUT_PATH = 'C:\\Development\\GraspNet\\graspnet-baseline\\logs\\my_logs\\train_3_mlp_approachnet_full\\model_pc_outputs'
DATASET_PATH = 'C:\\Development\\GraspNet\\graspnet-baseline\\dataset\\models'

OBJECT_LIST = list ( range(88) )
# OBJECT_LIST = [3, 4]
BATCH_SIZE = 1
COLLISION_THRESH = -1

TEST_DATASET = PCDataset(object_ids=OBJECT_LIST)
print(len(TEST_DATASET))

TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=0, collate_fn=collate_fn)

net = MyGraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                     cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint = torch.load(CHECKPOINT_PATH)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

def inference():
    batch_interval = 100
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    tic = time.time()
    for batch_idx, batch_data in enumerate(TEST_DATALOADER):
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        # Dump results for evaluation
        for i in range(BATCH_SIZE):
            data_idx = batch_idx * BATCH_SIZE + i
            preds = grasp_preds[i].detach().cpu().numpy()
            gg = GraspGroup(preds)

            # collision detection
            if COLLISION_THRESH > 0:
                # cloud, _ = TEST_DATASET.get_data(data_idx, return_raw_cloud=True)
                cloud = TEST_DATASET[data_idx]['point_clouds']
                mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
                gg = gg[~collision_mask]

            # save grasps
            save_dir = os.path.join(OUTPUT_PATH, str(OBJECT_LIST[data_idx]), 'kinect')
            save_path = os.path.join(save_dir, str(data_idx%256).zfill(4)+'.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            gg.save_npy(save_path)

        if batch_idx % batch_interval == 0:
            toc = time.time()
            print('Eval batch: %d, time: %fs'%(batch_idx, (toc-tic)/batch_interval))
            tic = time.time()
            
            
            
if __name__ == '__main__':
    inference()