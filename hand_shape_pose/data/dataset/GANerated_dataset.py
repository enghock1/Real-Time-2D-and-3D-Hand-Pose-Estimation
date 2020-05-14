"""
GANerated dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import scipy.io as sio
import os.path as osp
import logging
import cv2
import numpy as np
import numpy.linalg as LA
import math

import torch
import torch.utils.data

from hand_shape_pose.util.image_util import crop_pad_im_from_bounding_rect

SK_fx_color = 607.92271
SK_fy_color = 607.88192
SK_tx_color = 314.78337
SK_ty_color = 236.42484

GAN_fx = 617.173
GAN_fy = 617.173
GAN_tx = 315.453
GAN_ty = 242.259

GAN_rot = np.eye(3)
GAN_trans_vec = [ 24.7, -0.0471401, 3.72045]  # mm

GAN_joint_names = ['loc_bn_palm_L', 'loc_bn_thumb_L_01', 'loc_bn_thumb_L_02', 'loc_bn_thumb_L_03',
                    'loc_bn_thumb_L_04', 'loc_bn_index_L_01', 'loc_bn_index_L_02', 'loc_bn_index_L_03',
                    'loc_bn_index_L_04', 'loc_bn_mid_L_01', 'loc_bn_mid_L_02', 'loc_bn_mid_L_03',
                    'loc_bn_mid_L_04', 'loc_bn_ring_L_01', 'loc_bn_ring_L_02', 'loc_bn_ring_L_03',
                    'loc_bn_ring_L_04', 'loc_bn_pinky_L_01', 'loc_bn_pinky_L_02', 'loc_bn_pinky_L_03',
                    'loc_bn_pinky_L_04'
                    ]
GAN_joint_name2id = {w: i for i, w in enumerate(GAN_joint_names)}
resize_dim = [256, 256]


class GANDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir_list, image_prefix, bbox_file, ann_file_list):
        self.image_paths = []
        self.bboxes = []
        self.pose_roots = []
        self.pose_scales = []
        self.pose_gts = []
        self.cam_params = torch.tensor([GAN_fx, GAN_fy, GAN_tx, GAN_ty])

        root_id = GAN_joint_name2id['loc_bn_palm_L']

        for image_dir, ann_file in zip(image_dir_list, ann_file_list):
            mat_gt = sio.loadmat(ann_file)
            curr_pose_gts = mat_gt["handPara"].transpose((2, 1, 0))  # N x K x 3
            #curr_pose_gts = self.SK_xyz_depth2color(curr_pose_gts, GAN_trans_vec, GAN_rot)
            curr_pose_gts = curr_pose_gts / 10.0  # convert to Snap index, mm->cm
            #curr_pose_gts = self.palm2wrist(curr_pose_gts)  # N x K x 3
            curr_pose_gts = torch.from_numpy(curr_pose_gts)
            self.pose_gts.append(curr_pose_gts)

            self.pose_roots.append(curr_pose_gts[:, root_id, :])  # N x 3
            self.pose_scales.append(self.compute_hand_scale(curr_pose_gts))  # N

            for image_id in range(curr_pose_gts.shape[0]):
                self.image_paths.append(osp.join(image_dir, "%s_%d.png" % (image_prefix, image_id)))

        self.pose_roots = torch.cat(self.pose_roots, 0).float()
        self.pose_scales = torch.cat(self.pose_scales, 0).float()
        self.pose_gts = torch.cat(self.pose_gts, 0).float()


        mat_bboxes = sio.loadmat(bbox_file)
        self.bboxes = torch.from_numpy(mat_bboxes["bboxes"]).float()  # N x 4

    def __getitem__(self, index):
        img = cv2.imread(self.image_paths[index])
        '''crop image'''
        crop_img = crop_pad_im_from_bounding_rect(img, self.bboxes[index, :].int())
        '''resize image'''
        crop_resized_img = cv2.resize(crop_img, (resize_dim[1], resize_dim[0]))
        crop_resized_img = torch.from_numpy(crop_resized_img)  # 256 x 256 x 3

        return crop_resized_img, self.cam_params, self.bboxes[index], \
               self.pose_roots[index], self.pose_scales[index], index

    def __data_generation(self, dir_path):
        image = [np.asarray(Image.open(path+"_color_composed.png")) for path in dir_path]
        image = np.asarray(image, np.uint8)
        image = np.asarray(image, np.float)
        image = image / 255.0
        crop_param = []
        joint_3d = []
        joint_2d_heatmap = []
        joint_3d_rate = []
        for path in dir_path:
            value = open(path+"_crop_params.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            crop_param.append(value)

            value = open(path+"_joint_pos_global.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            joint_3d.append(value)

            value = open(path+"_joint_pos.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            joint_3d_rate.append(value)

            value = open(path+"_joint2D.txt").readline().strip('\n').split(',')
            value = [float(val) for val in value]
            value = np.asarray(value)
            value = np.reshape(value, (21, 2))
            for val in value:
                heat_map = gaussian_heat_map(val/8, self.heatmap_shape[0])
                joint_2d_heatmap.append(heat_map)

        crop_param = np.asarray(crop_param)
        crop_param = np.reshape(crop_param, (-1, 1, 3))

        joint_3d = np.asarray(joint_3d)
        joint_3d = np.reshape(joint_3d, (-1,63))

        joint_3d_rate = np.asarray(joint_3d_rate)
        joint_2d_heatmap = np.asarray(joint_2d_heatmap)
        joint_2d_heatmap = np.reshape(joint_2d_heatmap, (-1, 21, self.heatmap_shape[0], self.heatmap_shape[1]))
        joint_2d_heatmap = np.moveaxis(joint_2d_heatmap, 1, 3)

        return image, crop_param, joint_3d, joint_3d_rate, joint_2d_heatmap

    def make_dir_path(root_path):
        pathes = []
        no_object = root_path + "\\data\\noObject"
        for i in range(1,141):
            end = 1025
            if i == 69:
                end = 217
            for j in range(1,end):
                pathes.append(no_object +"\\{0:04d}\\{1:04d}".format(i,j))
        with_object = root_path + "\\data\\withObject"
        for i in range(1, 184):
            end = 1025
            if i == 92:
                end = 477
            for j in range(1, end):

                pathes.append(with_object + "\\{0:04d}\\{1:04d}".format(i, j))

        return pathes

    def __len__(self):
        return len(self.image_paths)

    def SK_xyz_depth2color(self, depth_xyz, trans_vec, rot_mx):
        """
        :param depth_xyz: N x 21 x 3, trans_vec: 3, rot_mx: 3 x 3
        :return: color_xyz: N x 21 x 3
        """
        color_xyz = depth_xyz - np.tile(trans_vec, [depth_xyz.shape[0], depth_xyz.shape[1], 1])
        return color_xyz.dot(rot_mx)

    def palm2wrist(self, pose_xyz):
        root_id = GAN_joint_name2id['loc_bn_palm_L']
        ring_root_id = GAN_joint_name2id['loc_bn_ring_L_01']
        pose_xyz[:, root_id, :] = pose_xyz[:, ring_root_id, :] + \
                                  2.0 * (pose_xyz[:, root_id, :] - pose_xyz[:, ring_root_id, :])  # N x K x 3
        return pose_xyz

    def compute_hand_scale(self, pose_xyz):
        ref_bone_joint_1_id = GAN_joint_name2id['loc_bn_mid_L_02']
        ref_bone_joint_2_id = GAN_joint_name2id['loc_bn_mid_L_01']

        pose_scale_vec = pose_xyz[:, ref_bone_joint_1_id, :] - pose_xyz[:, ref_bone_joint_2_id, :]  # N x 3
        pose_scale = torch.norm(pose_scale_vec, dim=1)  # N
        return pose_scale

    def evaluate_pose(self, results_pose_cam_xyz, save_results=False, output_dir=""):
        avg_est_error = 0.0
        for image_id, est_pose_cam_xyz in results_pose_cam_xyz.items():
            dist = est_pose_cam_xyz - self.pose_gts[image_id]  # K x 3
            avg_est_error += dist.pow(2).sum(-1).sqrt().mean()

        avg_est_error /= len(results_pose_cam_xyz)

        if save_results:
            eval_results = {}
            image_ids = results_pose_cam_xyz.keys()
            image_ids.sort()
            eval_results["image_ids"] = np.array(image_ids)
            eval_results["gt_pose_xyz"] = [self.pose_gts[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["est_pose_xyz"] = [results_pose_cam_xyz[image_id].unsqueeze(0) for image_id in image_ids]
            eval_results["gt_pose_xyz"] = torch.cat(eval_results["gt_pose_xyz"], 0).numpy()
            eval_results["est_pose_xyz"] = torch.cat(eval_results["est_pose_xyz"], 0).numpy()
            sio.savemat(osp.join(output_dir, "pose_estimations.mat"), eval_results)

        return avg_est_error.item()
