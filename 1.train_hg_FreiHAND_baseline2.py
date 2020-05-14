from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os.path as osp
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from hand_shape_pose.config import cfg
from hand_shape_pose.model.pose_mlp_network import MLPPoseNetwork
from hand_shape_pose.model.net_hg import Net_HM_HG
from hand_shape_pose.data.build import build_dataset

from hand_shape_pose.util.logger import setup_logger, get_logger_filename
from hand_shape_pose.util.miscellaneous import mkdir
from hand_shape_pose.util.vis import save_batch_image_with_mesh_joints
#from hand_shape_pose.util import renderer
from hand_shape_pose.util.net_util import load_net_model
from hand_shape_pose.util.image_util import BHWC_to_BCHW, normalize_image
import time

def main():
    parser = argparse.ArgumentParser(description="3D Hand Shape and Pose Inference")
    parser.add_argument(
        "--config-file",
        default="configs/train_FreiHAND_dataset.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 1. load network model
    num_joints = cfg.MODEL.NUM_JOINTS
    net_hm = Net_HM_HG(num_joints,
                       num_stages=cfg.MODEL.HOURGLASS.NUM_STAGES,
                       num_modules=cfg.MODEL.HOURGLASS.NUM_MODULES,
                       num_feats=cfg.MODEL.HOURGLASS.NUM_FEAT_CHANNELS)
    load_net_model(cfg.MODEL.PRETRAIN_WEIGHT.HM_NET_PATH, net_hm)
    device = cfg.MODEL.DEVICE
    net_hm.to(device)
    net_hm = net_hm.train()

    # 2. Load data

    dataset_val = build_dataset(cfg.TRAIN.DATASET, cfg.TRAIN.BACKGROUND_SET, cfg.TRAIN.DATA_SIZE)
    print('Perform dataloader...', end='')
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.MODEL.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.MODEL.NUM_WORKERS
    )
    print('done!')

    optimizer = optim.RMSprop(net_hm.parameters(), lr=10**-3)
    hm_loss = nn.MSELoss(reduction='sum')

    print('Entering loop...')
    num_epoch = 200
    for epoch in range(num_epoch):
        total_loss_train = 0.0
        tic = time.time()
        for i, batch in enumerate(data_loader_val):
            images, cam_params, pose_roots, pose_scales, image_ids = batch
            images, cam_params, pose_roots, pose_scales = \
                images.to(device), cam_params.to(device), pose_roots.to(device), pose_scales.to(device)

            # ground truth heatmap
            gt_heatmap = torch.Tensor().to(device)
            for img_id in image_ids:
                gt_heatmap = torch.cat((gt_heatmap, dataset_val.heatmap_gts_list[img_id].to(device)), 0)
            gt_heatmap = gt_heatmap.view(-1, 21, 64, 64)

            # backpropagation
            optimizer.zero_grad()
            images = BHWC_to_BCHW(images)  # B x C x H x W
            images = normalize_image(images)
            est_hm_list, _ = net_hm(images)
            est_hm_list = est_hm_list[-1].to(device)
            loss = hm_loss(est_hm_list, gt_heatmap)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()

        # record time
        toc = time.time()
        print('loss of epoch %2d: %6.2f, time: %0.4f s' %(int(epoch+1), total_loss_train, toc-tic))

        # save model weight every epoch
        torch.save(net_hm.state_dict(), "net_hm.pth")


if __name__ == "__main__":
    main()
