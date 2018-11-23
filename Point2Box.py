import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import time
import math
import KITTILoader as DA
import os


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


class Segment_net(nn.Module):
    def __init__(self, num_points):
        super(Segment_net, self).__init__()
        self.num_points = num_points
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.feature_net = nn.Sequential(
            convbn(1, 64, (1, 3), 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(64, 64, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(64, 128, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(128, 1024, 1, 1, 0, 1),
            nn.ReLU(inplace=True)
        )
        self.feature_net_max = nn.Sequential(
            nn.MaxPool2d([num_points, 1])
        )
        self.fc_net = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.segment_net = nn.Sequential(
            convbn(1024 + 128, 512, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(512, 256, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3),
            convbn(256, 128, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(128, 1, 1, 1, 0, 1)
        )

    def forward(self, points):
        new_points = points.unsqueeze(1)
        feature = self.feature_net(new_points)
        feature_max = self.feature_net_max(feature)
        global_feature = self.fc_net(feature_max.view(-1, 1024))
        new_size = [global_feature.size(0), global_feature.size(1), self.num_points, 1]
        expand_global_feature = global_feature.unsqueeze(-1).unsqueeze(-1).expand(new_size)
        cat_features = torch.cat((feature, expand_global_feature), dim=1)
        seg_mask = self.segment_net(cat_features)
        return seg_mask.squeeze(-1).squeeze(1)

    def getLoss(self, input, target):
        return self.loss(input, target)


class Center_regression_net(nn.Module):
    def __init__(self, num_points):
        super(Center_regression_net, self).__init__()
        self.loss = nn.SmoothL1Loss()
        self.feature_net = nn.Sequential(
            convbn(1, 128, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(128, 128, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(128, 256, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d([num_points, 1]))
        self.fc_net = nn.Sequential(
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, points):
        new_points = points.unsqueeze(1)
        features = self.feature_net(new_points)
        new_features = features.view(-1, 768)
        pred_center = self.fc_net(new_features)
        return pred_center

    def getLoss(self, predict, target):
        return self.loss(predict, target)


if __name__ == "__main__":
    test_load = torch.utils.data.DataLoader(
        DA.myPointData('obejct_data/data_object_image_2/training/frustum_points_train/', 1024, 12),
        batch_size=20, shuffle=True, num_workers=8, drop_last=False)

    # train 1 epoch for segment_net
    model = Segment_net(1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for batch_idx, (
            points,
            points_rot,
            seg_mask,
            box3d_center,
            box3d_center_rot,
            angle_c,
            angle_r,
            angle_c_rot,
            angle_r_rot,
            size_r) in enumerate(test_load):
        print(batch_idx)
        optimizer.zero_grad()
        out = model(points_rot)
        print(out.size())
        loss = model.getLoss(out, seg_mask)
        print(loss)
        loss.backward()
        optimizer.step()

    # train 1 epoch for center reg net
    model = Center_regression_net(512)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for batch_idx, (
            points,
            points_rot,
            seg_mask,
            box3d_center,
            box3d_center_rot,
            angle_c,
            angle_r,
            angle_c_rot,
            angle_r_rot,
            size_r) in enumerate(test_load):
        masked_points = sampleFromMask(points_rot, masked_points)
        optimizer.zero_grad()
        out = model(masked_points)
        loss = model.getLoss(out, box3d_center_rot)
        loss.backward()
        optimizer.step()
