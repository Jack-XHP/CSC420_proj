import numpy as np
import os
import cv2 as cv
from KITTIloader2015 import read_label, Object3d, read_2d_box
from scipy.spatial import Delaunay


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
    xmin, ymin, xmax, ymax = box2d
    h = ymax - ymin
    w = xmax - xmin
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    cx2 = cx + w * r * (np.random.random() * 2 - 1)
    cy2 = cy + h * r * (np.random.random() * 2 - 1)
    h2 = h * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    w2 = w * (1 + np.random.random() * 2 * r - r)  # 0.9 to 1.1
    return np.array([cx2 - w2 / 2.0, cy2 - h2 / 2.0, cx2 + w2 / 2.0, cy2 + h2 / 2.0])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def compute_box_3d(obj):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_3d: (8,3) array in in left camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l;
    w = obj.w;
    h = obj.h;

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2];
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h];
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2];

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0];
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1];
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2];
    return corners_3d.T


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def extract_frustum(points, img_id, index, point_dir, num_points, perturb_box2d=False, augmentX=1):
    label_file = 'obejct_data/data_object_image_2/training/label_2/{}.txt'.format(img_id)
    if os.path.isfile(label_file):
        objects = read_label(label_file)
    else:
        box_file = 'obejct_data/data_object_image_2/training/box/{}.txt'.format(img_id)
        objects = read_2d_box(box_file)
    image_x = points.shape[1] - 1
    image_y = points.shape[0] - 1
    for obj_idx in range(len(objects)):
        # 2D BOX: Get pts rect backprojected
        obj = objects[obj_idx]
        box2d = obj.box2d
        datas = {}
        box3d_count = []
        box3d_size = None
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
            ymax = min(image_y, ymax)
            point_2d = points[ymin: ymax + 1, xmin: xmax + 1]
            while point_2d.size / 3 > num_points:
                print(point_2d.shape)
                point_2d = np.delete(point_2d, list(range(0, point_2d.shape[0], 8)), axis=0)
                point_2d = np.delete(point_2d, list(range(0, point_2d.shape[1], 8)), axis=1)
            point_2d = point_2d.reshape((-1, 3))
            center_points = points[int(3 / 4.0 * ymin + ymax/4.0): int(3 / 4.0 * ymax + ymin/4.0) + 1, int(3 / 4.0 * xmin + xmax/4.0): int(3 / 4.0 * xmax + xmin/4.0) + 1]
            while center_points.size / 3 > num_points:
                print(center_points.shape)
                center_points = np.delete(center_points, list(range(0, center_points.shape[0], 8)), axis=0)
                center_points = np.delete(center_points, list(range(0, center_points.shape[1], 8)), axis=1)
            center_points = center_points.reshape((-1, 3))
            point_2d = np.vstack((point_2d, center_points))
            box2d_corner = np.array([xmin, ymin, xmax, ymax])
            box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0]).astype(int)
            center = points[(box2d_center[1], box2d_center[0])]
            frustum_angle = -1 * np.arctan2(center[2], center[0])
            if np.all(obj.t == 0):
                obj.t = center
            box3d_corner = compute_box_3d(obj)
            box3d_center = obj.t
            box3d_size = np.array([obj.l, obj.w, obj.h])
            _, inds = extract_pc_in_box3d(point_2d, box3d_corner)
            print(inds.sum())
            if inds.sum() < 1:
                print("skip")
                continue
            label = np.zeros(point_2d.shape[0])
            label[inds] = 1
            datas['img_id'] = img_id
            datas['point_2d'] = point_2d
            datas['box2d_corner'] = box2d_corner
            datas['box3d_corner'] = box3d_corner
            datas['box3d_center'] = box3d_center
            datas['box3d_size'] = box3d_size
            datas['frustum_angle'] = frustum_angle
            datas['heading'] = obj.ry
            datas['label'] = label
            np.save(point_dir + str(index), datas)
            datas = {}
            index += 1
        if box3d_size is not None:
            box3d_count.append(box3d_size)
    return index, box3d_count


def sampleFromMask_2(distribution, points, num_points):
    if distribution.sum() == 0:
        print("Errors: mask cannot be 0, will random sample")
        choice = np.random.choice(points.shape[0], num_points, replace=True)
        samples = points[choice, :]
        return samples
    if distribution.sum() != 1:
        distribution = distribution / distribution.sum()
    rand = np.random.rand(num_points)
    rand.sort()
    samples = []
    samplePos, distPos, cdf = 0, 0, distribution[0]
    while samplePos < num_points:
        if rand[samplePos] < cdf:
            samplePos += 1
            samples.append(points[distPos])
        else:
            distPos += 1
            if distPos > distribution.shape[0] - 1:
                cdf += 1
                distPos = distribution.shape[0] - 1
            else:
                cdf += distribution[distPos]
    return np.array(samples)





if __name__ == '__main__':
    dir = 'obejct_data/data_object_image_2/training/CNN_depth/'
    point_train = 'obejct_data/data_object_image_2/training/frustum_points_train/'
    point_val = 'obejct_data/data_object_image_2/training/frustum_points_val/'
    if not os.path.exists(point_train):
        os.makedirs(point_train)
    if not os.path.exists(point_val):
        os.makedirs(point_val)
    train_index = 0
    val_index = 0
    box3d_count = []
    for img in os.listdir(dir):
        points = find_3d_point(dir + img)
        img_id = img.split('.')[0]
        print(img_id)
        if int(img_id) < 700:
            train_index, box_count = extract_frustum(points, img_id, train_index, point_train, 1024, perturb_box2d=True,
                                                     augmentX=10)
            box3d_count = box3d_count + box_count
        else:
            val_index, box_count = extract_frustum(points, img_id, val_index, point_val, 1024, perturb_box2d=False,
                                                   augmentX=1)
            box3d_count = box3d_count + box_count
    np.savetxt('avg_box_size', np.mean(np.array(box3d_count), axis=0))
