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
from scipy.spatial import ConvexHull


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
    Args:
      subjectPolygon: a list of (x,y) 2d points, any polygon.
      clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
      **points have to be counter-clockwise ordered**
    Return:
      a list of (x,y) vertex point for the intersection polygon.
    """

    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])

    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]

    outputList = subjectPolygon
    cp1 = clipPolygon[-1]

    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]

        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0:
            return None
    return (outputList)


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def poly_area(x,y):
    """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c


def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0


def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0


def box3d_iou(corners1, corners2):
    ''' Compute 3D bounding box IoU.
    Input:
        corners1: numpy array (8,3), assume up direction is negative Y
        corners2: numpy array (8,3), assume up direction is negative Y
    Output:
        iou: 3D bounding box IoU
        iou_2d: bird's eye view 2D bounding box IoU
    todo (rqi): add more description on corner points' orders.
    '''
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)]
    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)
    ymax = min(corners1[0,1], corners2[0,1])
    ymin = max(corners1[4,1], corners2[4,1])
    inter_vol = inter_area * max(0.0, ymax-ymin)
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d


def get_box3d_corners(center, heading, size):
    l = size[:, 0].flatten()
    w = size[:, 1].flatten()
    h = size[:, 2].flatten()
    x_vector = torch.FloatTensor(np.array([1, 1, -1, -1, 1, 1, -1, -1]) * 0.5)
    y_vector = torch.FloatTensor(np.array([1, 1, 1, 1, -1, -1, -1, -1]) * 0.5)
    z_vector = torch.FloatTensor(np.array([1, -1, -1, 1, 1, -1, -1, 1]) * 0.5)

    if args.cuda:
        x_vector, y_vector, z_vector = x_vector.cuda, y_vector.cuda, z_vector.cuda

    x_corners = torch.ger(l, x_vector)
    y_corners = torch.ger(h, y_vector)
    z_corners = torch.ger(w, z_vector)
    corners = torch.cat((x_corners.unsqueeze(1), y_corners.unsqueeze(1), z_corners.unsqueeze(1)), dim=1)
    c = torch.cos(heading)
    s = torch.sin(heading)
    ones = torch.ones(heading.size(0))
    zeros = torch.zeros(heading.size(0))

    if args.cuda:
        c, s, ones, zeros = c.cuda, s.cuda, ones.cuda, zeros.cuda

    row1 = torch.stack((c, zeros, s), dim=1)
    row2 = torch.stack((zeros, ones, zeros), dim=1)
    row3 = torch.stack((-s, zeros, c), dim=1)
    R = torch.cat((row1.unsqueeze(1), row2.unsqueeze(1), row3.unsqueeze(1)), dim=1)

    corners_3d = torch.matmul(R, corners)
    corners_3d += center.unsqueeze(-1).repeat(1, 1, 8)
    corners_3d = torch.transpose(corners_3d, 1, 2)
    return corners_3d


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
            nn.Linear(256, 3 + 2 * num_head + 3)
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
        self.L1loss = nn.SmoothL1Loss()
        self.BCloss = nn.BCEWithLogitsLoss()
        self.CEloss = nn.CrossEntropyLoss()
        self.head_class = head_class
        self.mask_points = mask_points
        self.box_mean = torch.FloatTensor(
            np.array([4.031432506887061784e+00, 1.617190082644625493e+00, 1.517575757575760020e+00]))
        self.seg_model = Segment_net(num_points)
        self.center_model = Center_regression_net(mask_points)
        self.box_model = Box_3D_net(mask_points, head_class)

    def forward(self, points):
        # get masking of the input points
        pred_seg = self.seg_model(points)

        # sample points using mask as a distribution
        masked_points = []
        for i in range(seg_mask.size(0)):
            masked_points.append(sampleFromMask(pred_seg[i].detach().cpu().numpy(), points[i].cpu().numpy(), 512))
        masked_points = np.array(masked_points)

        # shift sampled pointed to their center
        masked_center = torch.FloatTensor(np.mean(masked_points, axis=1))
        masked_points = torch.FloatTensor(
            masked_points - np.repeat(np.expand_dims(masked_center, axis=1), self.mask_points, axis=1))

        if args.cuda:
            masked_center, masked_points = masked_center.cuda, masked_points.cuda

        # get a predict residual for box center
        pred_center_r = self.center_model(masked_points)

        # shift sampled points to box center
        masked_points = masked_points - pred_center_r.unsqueeze(1).expand(
            (pred_center_r.size(0), self.mask_points, pred_center_r.size(1)))

        # get box size, center and heading prediction
        box_pred = self.box_model(masked_points)

        return masked_center, pred_center_r, box_pred, pred_seg

    def getLoss(self, masked_center, pred_center_r, box_pred, pred_seg, seg_mask, box_center, head_c, head_r, size_r,
                box_weight=1.0, corner_weight=10.0, trace=False):
        target = torch.zeros(masked_center.size(0))
        target_corner = torch.zeros(masked_center.size(0), 8)
        head_c_onehot = torch.zeros(head_c.size(0), self.head_class)
        head_c_onehot.scatter_(1, head_c.unsqueeze(-1), 1)

        if args.cuda:
            target, target_corner,head_c_onehot = target.cuda, target_corner.cuda, head_c_onehot.cuda

        mask_loss = self.BCloss(pred_seg, seg_mask)

        center_loss = self.L1loss(torch.norm(pred_center_r + masked_center - box_center, dim=1), target)

        box_center_loss = self.L1loss(torch.norm(box_pred[:, :3]+pred_center_r + masked_center - box_center, dim=1), target)

        head_class_loss = self.CEloss(box_pred[:, 3:self.head_class + 3], head_c)

        head_residual_loss = self.L1loss(
            torch.sum(box_pred[:, self.head_class + 3:2 * self.head_class + 3] * head_c_onehot, dim=1),
            head_r / (np.pi / self.head_class))

        size_residual_loss = self.L1loss(box_pred[:, 2 * self.head_class + 3:], size_r / self.box_mean)

        pred_heading = head_c.type(torch.FloatTensor) * 2 * np.pi / self.head_class + torch.sum(
            box_pred[:, self.head_class + 3:2 * self.head_class + 3] * head_c_onehot, dim=1) * (np.pi / self.head_class)
        pred_size = box_pred[:, 2 * self.head_class + 3:] * self.box_mean + self.box_mean
        pred_corner_3d = get_box3d_corners(box_pred[:, :3], pred_heading, pred_size)

        heading = head_c.type(torch.FloatTensor) * 2 * np.pi / self.head_class + head_r
        flip_heading = heading + np.pi
        size = self.box_mean + size_r
        corner_3d = get_box3d_corners(box_center, heading, size)
        corner_3d_flip = get_box3d_corners(box_center, flip_heading, size)

        corner_dist = torch.min(torch.norm(pred_corner_3d - corner_3d, dim=-1), torch.norm(pred_corner_3d - corner_3d_flip, dim=-1))
        corner_loss = self.L1loss(corner_dist, target_corner)
        if trace:
            print(
                "  mask loss {}\n  center_loss {} \n  box_center_loss {} \n  head_class_loss {} \n  head_residual_loss {} \n  size_residual_loss {}\n  corner_loss {}".format(
                mask_loss, center_loss, box_center_loss, head_class_loss, head_residual_loss, size_residual_loss, corner_loss))

        return mask_loss + box_weight * (
                center_loss + box_center_loss + head_class_loss + head_residual_loss * 20
                + size_residual_loss * 20 + corner_weight * corner_loss), pred_corner_3d, corner_3d, corner_3d_flip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Point net')
    parser.add_argument('--datapath', default='depth_data/data/training/',
                        help='datapath')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--savemodel', default='./',
                        help='save model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--loadmodel', default=None,
                        help='load model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_load = torch.utils.data.DataLoader(
        DA.myPointData('obejct_data/data_object_image_2/training/frustum_points_train/', 1024, 12),
        batch_size=20, shuffle=True, num_workers=8, drop_last=False)
    valid_load = torch.utils.data.DataLoader(
        DA.myPointData('obejct_data/data_object_image_2/training/frustum_points_val/', 1024, 12),
        batch_size=1000, shuffle=False, num_workers=8, drop_last=False)

    # train 1 epoch for 3d box net
    model = Point_net(1024, 512, 12)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(args.epochs):
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
                size_r) in enumerate(train_load):
            points_rot = Variable(torch.FloatTensor(points_rot))
            seg_mask = Variable(torch.FloatTensor(seg_mask))
            box3d_center_rot = Variable(torch.FloatTensor(box3d_center_rot))
            angle_c_rot = Variable(torch.LongTensor(angle_c_rot.type(torch.LongTensor)))
            angle_r_rot = Variable(torch.FloatTensor(angle_r_rot.type(torch.FloatTensor)))
            size_r = Variable(torch.FloatTensor(size_r))

            if args.cuda:
                points_rot, seg_mask, box3d_center_rot, angle_c_rot, angle_r_rot, size_r = points_rot.cuda, seg_mask.cuda, box3d_center_rot.cuda, angle_c_rot.cuda, angle_r_rot.cuda, size_r.cuda

            optimizer.zero_grad()
            masked_center, pred_center_r, box_pred, pred_seg = model(points_rot)
            if batch_idx % 10 == 0:
                print("batch: {}".format(batch_idx))
                loss, pred_corner_3d, corner_3d, corner_3d_flip = model.getLoss(masked_center, pred_center_r, box_pred,
                                                                            pred_seg, seg_mask, box3d_center_rot,
                                                                            angle_c_rot,angle_r_rot, size_r, 1, 10, True)
                print("total loss {}".format(loss))
            else:
                loss, pred_corner_3d, corner_3d, corner_3d_flip = model.getLoss(masked_center, pred_center_r, box_pred,
                                                                                pred_seg, seg_mask, box3d_center_rot,
                                                                                angle_c_rot,angle_r_rot, size_r)

            loss.backward()
            optimizer.step()

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
                size_r) in enumerate(valid_load):
            points_rot = Variable(torch.FloatTensor(points_rot))
            seg_mask = Variable(torch.FloatTensor(seg_mask))
            box3d_center_rot = Variable(torch.FloatTensor(box3d_center_rot))
            angle_c_rot = Variable(torch.LongTensor(angle_c_rot.type(torch.LongTensor)))
            angle_r_rot = Variable(torch.FloatTensor(angle_r_rot.type(torch.FloatTensor)))
            size_r = Variable(torch.FloatTensor(size_r))

            if args.cuda:
                points_rot, seg_mask, box3d_center_rot, angle_c_rot, angle_r_rot, size_r = points_rot.cuda, seg_mask.cuda, box3d_center_rot.cuda, angle_c_rot.cuda, angle_r_rot.cuda, size_r.cuda
            masked_center, pred_center_r, box_pred, pred_seg = model(points_rot)
            loss, pred_corner_3d, corner_3d, corner_3d_flip = model.getLoss(masked_center, pred_center_r, box_pred,
                                                                            pred_seg, seg_mask, box3d_center_rot,
                                                                            angle_c_rot,
                                                                            angle_r_rot, size_r, 1, 10, True)

            print("valid total loss {}".format(loss))
            pred_corner_3d, corner_3d, corner_3d_flip = pred_corner_3d.detach().cpu().numpy(), corner_3d.detach().cpu().numpy(), corner_3d_flip.detach().cpu().numpy()
            IOU = []
            for i in range(corner_3d.shape[0]):
                IOU.append(box3d_iou(pred_corner_3d[i], corner_3d[i])[0])

            IOU = np.array(IOU)
            print(np.mean(IOU))
            print(IOU)
