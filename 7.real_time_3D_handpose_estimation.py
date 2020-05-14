"""
Real time 2D and 3D hand pose estimation using RGB webcam
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
import torch
import cv2

from hand_shape_pose.config import cfg
from hand_shape_pose.model.pose_mlp_network import MLPPoseNetwork
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis_pose_only import draw_2d_skeleton, draw_3d_skeleton

import matplotlib.pyplot as plt
import numpy as np

### specify inputs ###
config_file = "configs/eval_webcam.yaml"
K = [[291.00602819, 0, 139.59914484], [0, 292.75184403, 111.98793194], [0, 0, 1]]   # intrinsic camera parameter
pose_scale = 0.03           # hand pose scale ~3cm
cropped_dim = (480, 480)    # cropped dimension from the origial webcam image
resize_dim = (256, 256)     # input image dim accepted by the learning model
avg_per_frame = 1           # number of images averaged to help reduce noise

######################
 
cfg.merge_from_file(config_file)
cfg.freeze()

# Load trained network model
model = MLPPoseNetwork(cfg)
device = cfg.MODEL.DEVICE
model.to(device)
model.load_model(cfg, load_mlp=True)
model = model.eval()

# intrinsic camera parameter K and pose_scale
K = torch.tensor(K).to(device)
K = K.reshape((1,3,3))
pose_scale = torch.tensor(pose_scale).to(device).reshape((1,1)) # ~3cm

# webcam settings - default image size [640x480]
cap = cv2.VideoCapture(0)

# preset variables
ratio_dim = (cropped_dim[0]/resize_dim[0], cropped_dim[1]/resize_dim[1])
avg_est_pose_uv = np.zeros((21,2))
avg_est_pose_cam_xyz = np.zeros((21,3))
avg_frame = 0
azim_angle = -360
elev_angle = -360

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = frame[:,80:560,:]   # cut the frame to 480x480
    original_frame = frame.copy()
    frame = cv2.resize(frame, (resize_dim[1], resize_dim[0]))
    frame = frame.reshape((-1,resize_dim[1], resize_dim[0], 3))
    frame_device = torch.from_numpy(frame).to(device)
    
    # feed forward the model to obtain 2D and 3D hand pose
    _, est_pose_uv, est_pose_cam_xyz = model(frame_device, K, pose_scale)
    est_pose_uv = est_pose_uv.to('cpu')
    est_pose_cam_xyz = est_pose_cam_xyz.to('cpu')
    
    # shift est_pose_uv to calibrate pose position in the image
    est_pose_uv[0,:,0] = est_pose_uv[0,:,0]*ratio_dim[0]
    est_pose_uv[0,:,1] = est_pose_uv[0,:,1]*ratio_dim[1]

    # average hand pose with 3 frames to stabilize noise
    avg_est_pose_uv += est_pose_uv[0].detach().numpy()
    avg_est_pose_cam_xyz += est_pose_cam_xyz[0].detach().numpy()
    avg_frame += 1
    
    # Display the resulting frame
    if avg_frame == avg_per_frame:
        avg_frame = 0
        avg_est_pose_uv = avg_est_pose_uv/avg_per_frame + 25    # manual tuning to fit hand pose on top of the hand
        avg_est_pose_uv[:,1] += 10                              # same here

        # draw 2D hand pose
        skeleton_frame = draw_2d_skeleton(original_frame, avg_est_pose_uv)
        
        # rorate 3D hand pose
        azim_angle += 1
        elev_angle += 1
        if azim_angle == 360:
            azim_angle = -360
        if elev_angle == 360:
            elev_angle = -360
            
        # draw 3D hand pose
        skeleton_3D = draw_3d_skeleton(avg_est_pose_cam_xyz, cropped_dim, elev_angle=elev_angle, azim_angle=azim_angle)
        
        # plot hand poses
        cv2.imshow('hand pose estimation', np.concatenate((skeleton_frame,skeleton_3D[:,:,:3]),axis=1))
        avg_est_pose_cam_xyz = np.zeros((21,3))
        avg_est_pose_uv = np.zeros((21,2))
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # print frame per second    
    #fps = cap.get(cv2.CV_CAP_PROP_FPS)
    #print(fps," fps")

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
















