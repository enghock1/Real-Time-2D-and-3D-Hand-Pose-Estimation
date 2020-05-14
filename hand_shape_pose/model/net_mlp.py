from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from hand_shape_pose.model.net_hm_feat_mesh import Net_HM_Feat


class MLP(nn.Module):

	def __init__(self, num_heatmap_chan, num_feat_chan, size_input_feature=(64, 64)):
		super(MLP, self).__init__()

		self.feat_net = Net_HM_Feat(num_heatmap_chan, num_feat_chan, size_input_feature)
		self.bn1 = nn.BatchNorm1d(4096)
		self.drop1 = nn.Dropout(0.3)
		self.fc1 = nn.Linear(4096, 1024)
		self.bn2 = nn.BatchNorm1d(1024)
		self.drop2 = nn.Dropout(0.4)
		self.fc2 = nn.Linear(1024, 256)
		self.bn3 = nn.BatchNorm1d(256)
		self.drop3 = nn.Dropout(0.5)
		self.fc3 = nn.Linear(256, 20)

		nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
		nn.init.xavier_normal_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))

	def forward(self, hm_list, encoding_list):
		x = self.feat_net(hm_list, encoding_list)
		x = self.bn1(x)
		x = F.relu(x)
		x = self.drop1(x)
		x = self.fc1(x)
		x = self.bn2(x)
		x = F.relu(x)
		x = self.drop2(x)
		x = self.fc2(x)
		x = self.bn3(x)
		x = F.relu(x)
		x = self.drop3(x)
		x = self.fc3(x)

		return x

"""
    def __init__(self, num_heatmap_chan, num_feat_chan, size_input_feature=(64, 64)):
        super(MLP, self).__init__()
        self.feat_net = Net_HM_Feat(num_heatmap_chan, num_feat_chan, size_input_feature)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 20)

        nn.init.xavier_normal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc3.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, hm_list, encoding_list):
        x = self.feat_net(hm_list, encoding_list)
        #x = self.drop1(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.drop3(x)
        x = self.fc3(x)

        return x
"""
