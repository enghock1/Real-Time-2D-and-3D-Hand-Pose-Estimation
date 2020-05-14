# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Utilities for heat-map
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt

def find_keypoints_max(heatmaps):
  """
  heatmaps: C x H x W
  return: C x 3
  """
  # flatten the last axis
  heatmaps_flat = heatmaps.view(heatmaps.size(0), -1)

  # max loc
  max_val, max_ind = heatmaps_flat.max(1)
  max_ind = max_ind.float()

  max_v = torch.floor(torch.div(max_ind, heatmaps.size(1)))
  max_u = torch.fmod(max_ind, heatmaps.size(2))
  return torch.cat((max_u.view(-1,1), max_v.view(-1,1), max_val.view(-1,1)), 1)

def compute_uv_from_heatmaps(hm, resize_dim):
  """
  :param hm: B x K x H x W (Variable)
  :param resize_dim:
  :return: uv in resize_dim (Variable)
  """
  upsample = nn.Upsample(size=resize_dim, mode='bilinear')  # (B x K) x H x W
  resized_hm = upsample(hm).view(-1, resize_dim[0], resize_dim[1])

  uv_confidence = find_keypoints_max(resized_hm)  # (B x K) x 3

  return uv_confidence.view(-1, hm.size(1), 3)

def create_gaussian_heatmap_from_gt(uv_gts, uv_size=(224,224), hm_size=(64,64), std=4):
    heatmap_gt_list = []
    #heatmap_vis = np.zeros(hm_size)
    for uv_gt in uv_gts: # 21 joints
        u_max = uv_gt[0]*hm_size[0]/uv_size[0]  # column
        v_max = uv_gt[1]*hm_size[1]/uv_size[1]  # row
        gaussian_dist = multivariate_normal(mean=[u_max, v_max], cov=[[std**1, 0], [0, std**1]])
        heatmap_gt = np.zeros(hm_size)
        
        # get a 9x9 box with mean at the center
        up    = np.rint(v_max.numpy()-4) if v_max-4 >= 0 else 0
        down  = np.rint(v_max.numpy()+4) if v_max+4 <= hm_size[1] else hm_size[1]
        right = np.rint(u_max.numpy()+4) if u_max+4 <= hm_size[0] else hm_size[0]
        left  = np.rint(u_max.numpy()-4) if u_max-4 >= 0 else 0
        
        # create gaussian 
        for i in range(int(up),int(down)):
            for j in range(int(left),int(right)):
                heatmap_gt[i][j] = gaussian_dist.pdf([j, i])
        heatmap_gt = heatmap_gt/np.max(heatmap_gt) 
        #heatmap_vis = np.add(heatmap_vis, np.array(heatmap_gt))
        heatmap_gt_list.append(heatmap_gt.astype(np.float16))
        
    #plt.imshow(heatmap_vis)
    #plt.savefig("hm" + ".jpg")

    return heatmap_gt_list # 21 x 64 x 64




