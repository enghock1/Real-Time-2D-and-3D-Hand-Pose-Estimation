# Copyright (c) Liuhao Ge. All Rights Reserved.
r"""
Network utilities
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import torch
import numpy as np
import cmath



def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)


def normalize_image(im):
    """
    byte -> float, / pixel_max, - 0.5
    :param im: torch byte tensor, B x C x H x W, 0 ~ 255
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    """
    return ((im.float() / 255.0) - 0.5)


def denormalize_image(im):
    """
    float -> byte, +0.5, * pixel_max
    :param im: torch float tensor, B x C x H x W, -0.5 ~ 0.5
    :return:   torch byte tensor, B x C x H x W, 0 ~ 255
    """
    ret = (im + 0.5) * 255.0
    return ret.byte()


def uvd2xyz(uvd, cam_param, bbox, root_depth, pose_scale):
    """
    :param uvd: B x M x 3 (uvd is normalized between 0~1)
    :param cam_param: B x 4, [fx, fy, u0, v0]
    :param bbox: B x 4, bounding box in the original image, [x, y, w, h]
    :param root_depth: B
    :param pose_scale: B
    :return: mesh xyz coordinates in camera coordinate system, B x M x 3
    """
    '''1. denormalized uvd'''
    bbox = bbox.unsqueeze(1).expand(-1, uvd.size(1), -1)  # B x M x 4
    uv = uvd[:, :, :2] * bbox[:, :, 2:4] + bbox[:, :, :2]  # B x M x 2

    depth = uvd[:, :, 2] * pose_scale.unsqueeze(-1).expand_as(uvd[:, :, 2]) \
            + root_depth.unsqueeze(-1).expand_as(uvd[:, :, 2])  # B x M

    '''2. uvd->xyz'''
    cam_param = cam_param.unsqueeze(1).expand(-1, uvd.size(1), -1)  # B x M x 4
    xy = ((uv - cam_param[:, :, 2:4]) / cam_param[:, :, :2]) * depth.unsqueeze(-1).expand_as(uv)  # B x M x 2

    return torch.cat((xy, depth.unsqueeze(-1)), -1)  # B x M x 3

def uvd2xyz_freihand_train(uvd, cam_param, root_depth, pose_scale):
    """
    Reconstuct xyz position in camera coordinate given the root depth and hand scale
    :param uvd: B x M x 3 (uvd is normalized between 0~1)     B = batch, M = 20
    :param cam_param: 3 x 3
    :param root_depth: B
    :param pose_scale: B
    :return: mesh xyz coordinates in camera coordinate system, B x M x 3
    """

    """
    1. denormalized depth
    Denormalize the scale-invariant relative depth wrt root by multiplying hand scale
    then add the root depth to get the depth of all joints
    """
    depth = uvd[:, :, 2] * pose_scale.unsqueeze(-1).expand_as(uvd[:, :, 2]) \
            + root_depth.unsqueeze(-1).expand_as(uvd[:, :, 2])  # B x M

    '''2. uvd->xyz'''
    xyz_tensor = torch.zeros(uvd.size(0), 3, uvd.size(1))   # batch x 3 x 20
    for i in range(uvd.size(0)):
        #UVD = torch.cat((uvd[:, :, :2]*depth.unsqueeze(-1).expand_as(uvd[:, :, :2]), depth.unsqueeze(-1)), -1) # d*[u, v, 1]
        #xyz = torch.mm(torch.inverse(cam_param)[i, :, :], torch.transpose(UVD, 1, 2)[i, :, :]) # B x M x 3
        UVD = torch.cat((uvd[i, :, :2]*depth[i, :].unsqueeze(-1), depth[i, :].unsqueeze(-1)), -1) # d*[u, v, 1]
        xyz = torch.mm(torch.inverse(cam_param[i, :, :]), torch.transpose(UVD, 0, 1))
        xyz_tensor[i, :, :] = xyz
    xyz_tensor = torch.transpose(xyz_tensor, 1, 2)

    return xyz_tensor  # B x M x 3

def uvd2xyz_freihand_test(uvd, root_uv, cam_param, pose_scale):
    """
    Reconstruct xyz position in camera coordinate given 2.5D pose (uvd) and hand scale
    :param uvd: B x M x 3 (uvd is normalized between 0~1)
    :param cam_param: B x 4, [fx, fy, u0, v0]
    :param pose_scale: B
    :return: mesh xyz coordinates in camera coordinate system, B x M x 3
    """

    """
    1. Calculate normalized root depth
    Umar Iqbal,Pavlo Molchanov,Thomas Breuel Juergen Gall, and Jan Kautz.
    Hand pose estimation via latent 2.5 d heatmap regression.
    In Proc. of the Europ. Conf. on Computer Vision (ECCV), pages 118–134, 2018.
    """
    x9, y9, d9 = uvd[:, 8, 0], uvd[:, 8, 1], uvd[:, 8, 2] # joint 9: u, v, d
    x10, y10, d10 = uvd[:, 9, 0], uvd[:, 9, 1], uvd[:, 9, 2] # joint 10: u, v, d
    a = (x9 - x10)**2 + (y9 - y10)**2
    b = d9*(x9**2 + y9**2 - x9*x10 - y9*y10) + d10*(x10**2 + y10**2 - x9*x10 - y9*y10)
    c = (x9*d9 - x10*d10)**2 + (y9*d9 - y10*d10)**2 + (d9 - d10)**2 - 1
    
    device = c.device
    sqrt_term = torch.tensor([cmath.sqrt(b[i]**2 - 4*a[i]*c[i]).imag for i in range(a.shape[0])]).to(device)
    
    d_root = (0.5*(- b + sqrt_term)/a).unsqueeze(1) # B x 1 (scale-normalized depth of root joint)
    
    uvd[:, :, 2] = uvd[:, :, 2] + d_root.expand_as(uvd[:, :, 2])
    uvd_root = torch.cat((root_uv, d_root), -1) # B x 3
    uvd_with_root = torch.cat((uvd_root.unsqueeze(1), uvd), 1) # B x K x 3
    uvd_with_root[:, :, :2] = uvd_with_root[:, :, :2]*uvd_with_root[:, :, 2].unsqueeze(-1).expand_as(uvd_with_root[:, :, :2]) # [u, v, d] -> d*[u, v, 1]    
    
    '''2. uvd->xyz'''
    xyz_tensor_with_root = torch.zeros(uvd.size(0), 3, 21)
    for i in range(uvd.size(0)):
        xyz_with_root = torch.mm(torch.inverse(cam_param)[i, :, :], torch.transpose(uvd_with_root, 1, 2)[i, :, :]) # B x K x 3
        xyz_with_root = pose_scale[i]*xyz_with_root # denormalize
        xyz_tensor_with_root[i, :, :] = xyz_with_root
    xyz_tensor_with_root = torch.transpose(xyz_tensor_with_root, 1, 2)

    return xyz_tensor_with_root  # B x K x 3

def bounding_rect_pts(pts):
    """
    get bounding box of pts
    :param pts: N x K x 2
    :return: N x 4, [x, y, w, h]
    """
    pt_max, _ = torch.max(pts, 1)  # N x 2
    pt_min, _ = torch.min(pts, 1)  # N x 2

    return torch.cat((pt_min, pt_max - pt_min + 1), 1)  # N x 4


def pad_bounding_rect(bbox, pad_sz, image_shape):
    """
    :param bbox: N x 4, [x, y, w, h]
    :param pad_sz
    :param image_shape: [H, W]
    :return: N x 4, [x, y, w, h]
    """
    x_upper = bbox[:, 0] + bbox[:, 2]  # N
    y_upper = bbox[:, 1] + bbox[:, 3]  # N

    xy_pad = torch.clamp(bbox[:, :2] - pad_sz, min=0.0)  # N x 2

    w_pad = torch.min(x_upper - xy_pad[:, 0] + pad_sz, image_shape[1] - xy_pad[:, 0])  # N
    h_pad = torch.min(y_upper - xy_pad[:, 1] + pad_sz, image_shape[0] - xy_pad[:, 1])  # N

    return torch.cat((xy_pad, w_pad.unsqueeze(-1), h_pad.unsqueeze(-1)), 1)  # N x 4


def expand_bounding_rect(bbox, image_dim, resize_dim):
    """
    :param bbox: [x, y, w, h]
    :param image_dim: [H, W]
    :param resize_dim: [H_r, W_r]
    :return: N x 4, [x, y, w, h]
    """
    place_ratio = 0.5
    bbox_expand = bbox
    if resize_dim[0] / bbox[3] > resize_dim[1] / bbox[2]:  # keep width
        bbox_expand[3] = resize_dim[0] * bbox[2] / resize_dim[1]
        bbox_expand[1] = max(min(bbox[1] - (bbox_expand[3] - bbox[3]) * place_ratio,
                                 image_dim[0] - bbox_expand[3]), 0.0)
    else:  # keep height
        bbox_expand[2] = resize_dim[1] * bbox[3] / resize_dim[0]
        bbox_expand[0] = max(min(bbox[0] - (bbox_expand[2] - bbox[2]) * place_ratio,
                                 image_dim[1] - bbox_expand[2]), 0.0)

    return bbox_expand.int()


def crop_pad_im_from_bounding_rect(im, bb):
    """
    :param im: H x W x C
    :param bb: x, y, w, h (may exceed the image region)
    :return: cropped image
    """
    crop_im = im[max(0, bb[1]):min(bb[1] + bb[3], im.shape[0]), max(0, bb[0]):min(bb[0] + bb[2], im.shape[1]), :]

    if bb[1] < 0:
        crop_im = cv2.copyMakeBorder(crop_im, -bb[1], 0, 0, 0,  # top, bottom, left, right, bb[3]-crop_im.shape[0]
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    if bb[1] + bb[3] > im.shape[0]:
        crop_im = cv2.copyMakeBorder(crop_im, 0, bb[1] + bb[3] - im.shape[0], 0, 0,
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    if bb[0] < 0:
        crop_im = cv2.copyMakeBorder(crop_im, 0, 0, -bb[0], 0,  # top, bottom, left, right
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    if bb[0] + bb[2] > im.shape[1]:
        crop_im = cv2.copyMakeBorder(crop_im, 0, 0, 0, bb[0] + bb[2] - im.shape[1],
                                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
    return crop_im
