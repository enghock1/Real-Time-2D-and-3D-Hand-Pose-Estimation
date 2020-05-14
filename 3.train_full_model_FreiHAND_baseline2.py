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

    # Load network model
    model = MLPPoseNetwork(cfg)
    device = cfg.MODEL.DEVICE
    model.to(device)
    model.load_model(cfg, load_mlp=True)
    model = model.train()

    # Load data
    dataset_val = build_dataset(cfg.TRAIN.DATASET, cfg.TRAIN.BACKGROUND_SET, cfg.TRAIN.DATA_SIZE)
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=cfg.MODEL.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.MODEL.NUM_WORKERS
    )

    # set optimizer
    optimizer = optim.RMSprop(model.parameters(), lr=10**-4)
    hm_loss = nn.MSELoss(reduction='sum')
    pose_loss = nn.MSELoss(reduction='sum')

    num_epoch = 400
    for epoch in range(num_epoch):

        # reduce learning rate every 50 epoch
        if epoch % 50 == 0 and epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10

        tic = time.time()
        total_loss_train = 0.0

        # model training per batch
        for i, batch in enumerate(data_loader_val):
            images, cam_params, pose_roots, pose_scales, image_ids = batch
            images, cam_params, pose_roots, pose_scales = \
                images.to(device), cam_params.to(device), pose_roots.to(device), pose_scales.to(device)

            # ground truth heatmap
            gt_heatmap = torch.Tensor().to(device)
            for img_id in image_ids:
                gt_heatmap = torch.cat((gt_heatmap, dataset_val.heatmap_gts_list[img_id].to(device)), 0)
            gt_heatmap = gt_heatmap.view(-1, 21, 64, 64)

            # ground truth pose
            gt_pose_cam_xyz = torch.Tensor().to(device)
            for img_id in image_ids:
                gt_pose_cam_xyz = torch.cat((gt_pose_cam_xyz, dataset_val.pose_gts[img_id].to(device)), 0)
            gt_pose_cam_xyz = gt_pose_cam_xyz.view(-1, 21, 3)

            # obtain estimate and loss, then backpropogate
            optimizer.zero_grad()
            est_hm_list, est_pose_uv, est_pose_cam_xyz = model(images, cam_params, pose_scales, pose_root=pose_roots)
            loss = hm_loss(est_hm_list, gt_heatmap) + pose_loss(est_pose_cam_xyz, gt_pose_cam_xyz)
            loss.backward()
            optimizer.step()
            total_loss_train += loss.item()

        toc = time.time()
        print('loss of epoch %2d: %6.2f, time: %0.4f s' %(int(epoch+1), total_loss_train, toc-tic))

        # save model weight every epoch
        torch.save(model.net_hm.state_dict(), "net_hm.pth")
        torch.save(model.mlp.state_dict(), "mlp.pth")


if __name__ == "__main__":
    main()
