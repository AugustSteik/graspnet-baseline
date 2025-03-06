NOTE: Currently missing scenes 100-189, Test Images section on website. 

Dataset structure:
|-- dataset
    |-- scenes
    |   |-- scene_0000/
    |   |-- scene_0001/
    |   |-- ... ...
    |   `-- scene_0189/
    |
    |
    |-- models
    |   |-- 000/
    |   |-- 001/
    |   |-- ...
    |   `-- 087/
    |
    |
    |-- dex_models(optional but strongly recommended for accelerating evaluation)
    |   |-- 000.pkl
    |   |-- 001.pkl
    |   |-- ...
    |   `-- 087.pkl
    |   
    |
    |-- grasp_label
    |   |-- 000_labels.npz
    |   |-- 001_labels.npz
    |   |-- ...
    |   `-- 087_labels.npz
    |
    |
    `-- collision_label
        |-- scene_0000/
        |-- scene_0001/
        |-- ... ...
        `-- scene_0189/


2. Detail structure of each scene
|-- scenes
    |-- scene_0000
    |   |-- object_id_list.txt              # objects' id that appear in this scene, 0-indexed
    |   |-- rs_wrt_kn.npy                   # realsense camera pose with respect to kinect, shape: 256x(4x4)
    |   |-- kinect                          # data of kinect camera
    |   |   |-- rgb                         
    |   |   |   |-- 0000.png to 0255.png    # 256 rgb images
    |   |   `-- depth
    |   |   |   |-- 0000.png to 0255.png    # 256 depth images
    |   |   `-- label
    |   |   |   |-- 0000.png to 0255.png    # 256 object mask images, 0 is background, 1-88 denotes each object (1-indexed), same format as YCB-Video dataset
    |   |   `-- annotations
    |   |   |   |-- 0000.xml to 0255.xml    # 256 object 6d pose annotation. ‘pos_in_world' and'ori_in_world' denotes position and orientation w.r.t the camera frame. 
    |   |   `-- meta
    |   |   |   |-- 0000.mat to 0255.mat    # 256 object 6d pose annotation, same format as YCB-Video dataset for easy usage
    |   |   `-- rect
    |   |   |   |-- 0000.npy to 0255.npy    # 256 2D planar grasp labels
    |   |   |   
    |   |   `-- camK.npy                    # camera intrinsic, shape: 3x3, [[f_x,0,c_x], [0,f_y,c_y], [0,0,1]]
    |   |   `-- camera_poses.npy            # 256 camera poses with respect to the first frame, shape: 256x(4x4)
    |   |   `-- cam0_wrt_table.npy          # first frame's camera pose with respect to the table, shape: 4x4
    |   |
    |   `-- realsense
    |       |-- same structure as kinect
    |
    |
    `-- scene_0001
    |
    `-- ... ...
    |
    `-- scene_0189