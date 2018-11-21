import numpy as np
import os
import cv2 as cv
from KITTIloader2015 import read_label, Object3d, read_2d_box


class Voxel():
    def __init__(self, size):
        self.voxels = None
        self.min_coor = None
        self.max_coor = None
        self.num_division = None
        self.voxels_coor = None
        self.free_space = None
        self.size = size

    def discrete_points(self, points):
        voxels = np.floor_divide(points, self.size).astype(int)
        self.voxels_coor, voxel_index = np.unique(voxels, return_index=True, axis=0)
        self.voxels = points[voxel_index]
        self.min_coor = np.amin(self.voxels_coor, axis=0)
        self.min_coor[2] = max(self.min_coor[2] - np.floor_divide(1, self.size), 0)
        self.max_coor = np.amax(self.voxels_coor, axis=0)
        self.max_coor[1] += np.floor_divide(0.5, self.size)
        self.num_division = (self.max_coor - self.min_coor + 1).astype(int)
        occupancy_id = (self.voxels_coor - self.min_coor)
        self.free_space = -1 * np.ones(tuple(self.num_division))
        self.free_space[occupancy_id[:, 0], occupancy_id[:, 1], occupancy_id[:, 2]] = 1


def find_3d_point(disp_path):
    f = 721.537700
    px = 609.559300
    py = 172.854000
    T = 0.5327119288
    disp = cv.imread(disp_path, 0)
    depth = f * T / disp
    index = np.indices(disp.shape)
    x = (index[1] - px) / f * depth
    y = (index[0] - py) / f * depth
    points = np.vstack((x.flatten(), y.flatten(), depth.flatten())).T
    mask = depth.flatten() > 1000
    points[mask, ...] = np.zeros(3)
    return points.reshape((disp.shape[0], disp.shape[1], 3))

def random_shift_box2d(box2d, shift_ratio=0.1):
    ''' Randomly shift box center, randomly scale width and height
    '''
    r = shift_ratio
    xmin,ymin,xmax,ymax = box2d
    h = ymax-ymin
    w = xmax-xmin
    cx = (xmin+xmax)/2.0
    cy = (ymin+ymax)/2.0
    cx2 = cx + w*r*(np.random.random()*2-1)
    cy2 = cy + h*r*(np.random.random()*2-1)
    h2 = h*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    w2 = w*(1+np.random.random()*2*r-r) # 0.9 to 1.1
    return np.array([cx2-w2/2.0, cy2-h2/2.0, cx2+w2/2.0, cy2+h2/2.0])

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def extract_frustum(points, img_id, perturb_box2d=False, augmentX=1):
    label_file = 'obejct_data/data_object_image_2/training/label_2/{}.txt'.format(img_id)
    if os.path.isfile(label_file):
        objects = read_label(label_file)
    else:
        box_file = 'obejct_data/data_object_image_2/training/box/{}.txt'.format(img_id)
        objects = read_2d_box(box_file)
    box2d_list = []  # [xmin,ymin,xmax,ymax]
    box3d_list = []  # (8,3) array in rect camera coord
    input_list = []  # channel number = 4, xyz,intensity in rect camera coord
    label_list = []  # 1 for roi object, 0 for clutter
    heading_list = []  # ry (along y-axis in rect camera coord) radius of
    # (cont.) clockwise angle from positive x axis in velo coord.
    box3d_size_list = []  # array of l,w,h
    frustum_angle_list = []  # angle of 2d box center from pos x-axi
    image_x = points.shape[1]-1
    image_y = points.shape[0]-1
    for obj_idx in range(len(objects)):
        # 2D BOX: Get pts rect backprojected
        obj = objects[obj_idx]
        box2d = obj.box2d
        for i in range(augmentX):
            # Augment data by box2d perturbation
            if perturb_box2d:
                xmin, ymin, xmax, ymax = random_shift_box2d(box2d).astype(int)
            else:
                xmin, ymin, xmax, ymax = box2d.astype(int)
            if ymax - ymin < 25 or obj.type != 'Car' or xmax - xmin < 10:
                continue
            xmin = max(0, xmin)
            xmax = min(image_x, xmax)
            ymin = max(0, ymin)
            ymax = min(image_y,  ymax)
            point_2d = points[ymin : ymax + 1, xmin : xmax + 1].reshape((-1,3))
            box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0]).astype(int)
            center = points[(box2d_center[1], box2d_center[0])]
            frustum_angle = -1 * np.arctan2(center[2], center[0])
            label = np.zeros((points.shape[0], points.shape[1]))
            label[ymin : ymax + 1, xmin : xmax + 1] = 1
            label = label.flatten()
            box3d_center = obj.t
            if np.all(box3d_center == 0):
                box3d_center = center
            box3d_size = np.array([obj.l, obj.w, obj.h])
            box2d_list.append(np.array([xmin, ymin, xmax, ymax]))
            box3d_list.append(box3d_center)
            input_list.append(point_2d)
            label_list.append(label)
            heading_list.append(obj.ry)
            box3d_size_list.append(box3d_size)
            frustum_angle_list.append(frustum_angle)
    return box2d_list, box3d_list, input_list, label_list, heading_list, box3d_size_list, frustum_angle_list




if __name__ == '__main__':
    dir = 'obejct_data/data_object_image_2/training/CNN_depth/'
    point_dir = 'obejct_data/data_object_image_2/training/frustum_points/'
    if not os.path.exists(point_dir):
        os.makedirs(point_dir)
    id_list = []
    for img in os.listdir(dir):
        points = find_3d_point(dir + img)
        index = img.split('.')[0]
        print(index)
        if int(index) < 700:
            box2d_list, box3d_list, input_list, label_list, heading_list, \
                box3d_size_list, frustum_angle_list = extract_frustum(points, index, perturb_box2d=True, augmentX=5)

        else:
            box2d_list, box3d_list, input_list, label_list, heading_list, \
            box3d_size_list, frustum_angle_list = extract_frustum(points, index, perturb_box2d=False, augmentX=1)
        np.save('obejct_data/data_object_image_2/training/frustum_points/id_list_{}'.format(index), id_list)
        np.save('obejct_data/data_object_image_2/training/frustum_points/box2d_list_{}'.format(index), box2d_list)
        np.save('obejct_data/data_object_image_2/training/frustum_points/box3d_list_{}'.format(index), box3d_list)
        np.save('obejct_data/data_object_image_2/training/frustum_points/input_list_{}'.format(index), input_list)
        np.save('obejct_data/data_object_image_2/training/frustum_points/label_list_{}'.format(index), label_list)
        np.save('obejct_data/data_object_image_2/training/frustum_points/heading_list_{}'.format(index), heading_list)
        np.save('obejct_data/data_object_image_2/training/frustum_points/box3d_size_list_{}'.format(index), box3d_size_list)
        np.save('obejct_data/data_object_image_2/training/frustum_points/frustum_angle_list_{}'.format(index), frustum_angle_list)

