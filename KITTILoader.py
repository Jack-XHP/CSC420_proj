"""
adapted from https://github.com/JiaRenChang/PSMNet for
"Pyramid Stereo Matching Network" paper (CVPR 2018) by Jia-Ren Chang and Yong-Sheng Chen.
"""
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import os

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

__box_mean = [4.031432506887061784e+00, 1.617190082644625493e+00, 1.517575757575760020e+00]

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]

    return transforms.Compose(t_list)


def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None):
    normalize = __imagenet_stats
    input_size = 256
    return scale_crop(input_size=input_size,
                      scale_size=scale_size, normalize=normalize)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return Image.open(path)


def point_loader(path):
    return np.load(path).item()


def rotate_pc_along_y(pc, rot_angle):
    '''
    Input:
        pc: numpy array (N,C), first 3 channels are XYZ
            z is facing forward, x is left ward, y is downward
        rot_angle: rad scalar
    Output:
        pc: updated pc with XYZ rotated
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class and residual.
    Input:
        angle: rad scalar, from 0-2pi (or -pi~pi), class center at
            0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        num_class: int scalar, number of classes N
    Output:
        class_id, int, among 0,1,...,N-1
        residual_angle: float, a number such that
            class*(2pi/N) + residual_angle = angle
    '''
    angle = angle % (2 * np.pi)
    assert (angle >= 0 and angle <= 2 * np.pi)
    angle_per_class = 2 * np.pi / float(num_class)
    shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
    class_id = int(shifted_angle / angle_per_class)
    residual_angle = shifted_angle - \
                     (class_id * angle_per_class + angle_per_class / 2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class, to_label_format=True):
    ''' Inverse function to angle2class.
    If to_label_format, adjust angle to the range as in labels.
    '''
    angle_per_class = 2 * np.pi / float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    if to_label_format and angle > np.pi:
        angle = angle - 2 * np.pi
    return angle


def size2class(size):
    ''' Convert 3D bounding box size to template class and residuals.
    todo (rqi): support multiple size clusters per type.

    Input:
        size: numpy array of shape (3,) for (l,w,h)
    Output:
        size_residual: numpy array of shape (3,)
    '''
    size_residual = size - __box_mean
    return size_residual


def class2size(residual):
    ''' Inverse function to size2class. '''
    mean_size = __box_mean
    return mean_size + residual


class myPointData(data.Dataset):
    def __init__(self, points_dir, num_point, num_angle):
        self.points = [points_dir + point for point in os.listdir(points_dir)]
        self.num_point = num_point
        self.num_angle = num_angle

    def __getitem__(self, index):
        point = self.points[index]
        datas = point_loader(point)
        rot_angle = np.pi / 2 + datas['frustum_angle']
        points = datas['point_2d']
        # sampling n points from whole point cloud
        choice = np.random.choice(points.shape[0], self.num_point, replace=True)
        points = points[choice, :]

        # find the mask label for 3d points
        seg_mask = datas['label'][choice]

        # get 3d box center
        box3d_corner = datas['box3d_corner']
        box3d_center = (box3d_corner[0, :] + box3d_corner[6, :]) / 2.0

        # convert heading to 12 classes and residual
        head = datas['heading']
        angle_c, angle_r = angle2class(head, self.num_angle)
        # convert 3d box size to mean + residual
        size_r = size2class(datas['box3d_size'])

        # rotate points and boxes to center of frustum
        points_rot = rotate_pc_along_y(points.copy(), rot_angle)
        box3d_center_rot = rotate_pc_along_y(np.expand_dims(box3d_center, 0), rot_angle).squeeze()
        angle_c_rot, angle_r_rot = angle2class(head - rot_angle, self.num_angle)

        return points, points_rot, box3d_center, box3d_center_rot, angle_c, angle_r, angle_c_rot, angle_r_rot, size_r

    def __len__(self):
        return len(self.points)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, name, load=False, loader=default_loader,
                 dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training
        self.name = name
        self.load = load

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]

        name = self.name[index]

        left_img = self.loader(left)
        right_img = self.loader(right)

        if not self.load:
            disp_L = self.disp_L[index]
            dataL = self.dploader(disp_L)

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            if not self.load:
                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
                dataL = dataL[y1:y1 + th, x1:x1 + tw]
            else:
                dataL = 0
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL, name
        else:
            w, h = left_img.size

            left_img = left_img.crop((w - 1232, h - 368, w, h))
            right_img = right_img.crop((w - 1232, h - 368, w, h))
            w1, h1 = left_img.size
            if not self.load:
                dataL = dataL.crop((w - 1232, h - 368, w, h))
                dataL = np.ascontiguousarray(dataL, dtype=np.float32) / 256
            else:
                dataL = 0

            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL, name

    def __len__(self):
        return len(self.left)


if __name__ == '__main__':
    test_load = torch.utils.data.DataLoader(
        myPointData('obejct_data/data_object_image_2/training/frustum_points_train/', 1024, 12),
        batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    for batch_idx, (points, points_rot, box3d_center, box3d_center_rot, angle_c, angle_r, angle_c_rot, angle_r_rot,
                    size_r) in enumerate(test_load):
        print("batch:{}".format(batch_idx))
        print(points)
        print(points_rot)
        print(box3d_center)
        print(box3d_center_rot)
        print(angle_c)
        print(angle_r)
        print(angle_c_rot)
        print(angle_r_rot)
        print(size_r)
