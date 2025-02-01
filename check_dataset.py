__author__ = 'mhgou'
__version__ = '1.0'

import os
from graspnetAPI import GraspNet

# GraspNetAPI example for checking the data completeness.
# change the graspnet_root path

if __name__ == '__main__':

    ####################################################################
    graspnet_root = os.path.join(os.path.dirname(__file__), 'dataset')  ### ROOT PATH FOR GRASPNET ###
    ####################################################################

    g = GraspNet(graspnet_root, 'kinect', 'all')
    if g.checkDataCompleteness():
        print('Check for kinect passed')


    g = GraspNet(graspnet_root, 'realsense', 'all')
    if g.checkDataCompleteness():
        print('Check for realsense passed')
        
        