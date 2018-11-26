import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
import KITTILoader as DA
from Compute_3D_point import sampleFromMask
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
            convbn(128, 128, 1, 1, 0, 1),
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


class Box_3D_net(nn.Module):
    def __init__(self, num_points, num_head):
        super(Box_3D_net, self).__init__()
        self.feature_net = nn.Sequential(
            convbn(1, 128, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(128, 128, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(128, 256, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            convbn(256, 512, 1, 1, 0, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d([num_points, 1])
        )
        self.fc_net = nn.Sequential(
            nn.Linear(1536, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 3+2*num_head+3)
        )

    def forward(self, points):
        new_points = points.unsqueeze(1)
        feature = self.feature_net(new_points)
        predict = self.fc_net(feature.view(-1, 1536))
        return predict


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


class Point_net(nn.Module):
    def __init__(self, num_points, mask_points, head_class):
        super(Point_net, self).__init__()
        self.mask_points = mask_points
        self.seg_model = Segment_net(num_points)
        self.center_model = Center_regression_net(mask_points)
        self.box_model = Box_3D_net(mask_points, head_class)

    def forward(self, points):
        # get masking of the input points
        pred_seg = self.seg_model(points)

        # sample points using mask as a distribution
        masked_points = []
        for i in range(seg_mask.size(0)):
            masked_points.append(sampleFromMask(pred_seg[i].detach().numpy(), points[i].numpy(), 512))
        masked_points = np.array(masked_points)

        # shift sampled pointed to their center
        masked_center = np.mean(masked_points, axis=1)
        masked_points = torch.FloatTensor(masked_points - np.repeat(np.expand_dims(masked_center, axis=1), self.mask_points, axis=1))

        # get a predict residual for box center
        pred_center_r = self.center_model(masked_points)

        # shift sampled points to box center
        masked_points -= pred_center_r.unsqueeze(1).expand((pred_center_r.size(0), self.mask_points, pred_center_r.size(1)))

        # get box size, center and heading prediction
        box_pred = self.box_model(masked_points)

        return masked_center, pred_center_r, box_pred, pred_seg

    
if __name__ == "__main__":
    test_load = torch.utils.data.DataLoader(
        DA.myPointData('obejct_data/data_object_image_2/training/frustum_points_train/', 1024, 12),
        batch_size=20, shuffle=True, num_workers=8, drop_last=False)
    '''
    # train 1 epoch for segment_net
    model = Segment_net(1024)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
        masked_points = []
        for i in range(seg_mask.size(0)):
            masked_points.append(sampleFromMask(seg_mask[i].numpy(), points_rot[i].numpy() ,512))
        masked_points = torch.FloatTensor(np.array(masked_points))
        optimizer.zero_grad()
        out = model(masked_points)
        loss = model.getLoss(out, box3d_center_rot)
        print("batch {}, loss{}".format(batch_idx, loss))
        loss.backward()
        optimizer.step()
    '''
    # train 1 epoch for 3d box net
    model = Point_net(1024, 512, 12)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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
        optimizer.zero_grad()
        out = model(points_rot)
        optimizer.step()
