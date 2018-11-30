import numpy as np
import os
import cv2 as cv
from KITTIloader2015 import read_label, Object3d, read_2d_box
from scipy.spatial import Delaunay
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
<<<<<<< HEAD
=======

>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276

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
    T = 0.5327119288
    disp = cv.imread(disp_path, 0)
    depth = f * T / disp
    index = np.indices(disp.shape)
    x = index[1]
    y = index[0]
    points = np.vstack((x.flatten(), y.flatten(), depth.flatten())).T
    mask = depth.flatten() > 100
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
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    #print corners_3d.shape
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0]
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1]
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2]
    return np.transpose(corners_3d)


def in_hull(p, hull):
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds

def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line)==0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

class Calib(object):
    def __init__(self, calib_filepath):
        calibs = read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative
        self.b_y = self.P[1,3]/(-self.f_v)

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom

    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
<<<<<<< HEAD

    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)


    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_image_to_rect(self, uv_depth):
=======

    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)


    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    def project_image_to_rect(self, uv_depth):
=======

def project_image_to_rect(calib, uv_depth):
>>>>>>> e8b96137bc6f28c671e46343aa6b0540a27e1356
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
<<<<<<< HEAD
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
=======
<<<<<<< HEAD
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
=======
        x = ((uv_depth[:,0]-calib.c_u)*uv_depth[:,2])/calib.f_u + calib.b_x
        y = ((uv_depth[:,1]-calib.c_v)*uv_depth[:,2])/calib.f_v + calib.b_y
>>>>>>> e8b96137bc6f28c671e46343aa6b0540a27e1356
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect

<<<<<<< HEAD

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def extract_frustum(path, points, img_id, index, point_dir, num_points, perturb_box2d=False, augmentX=1):
    image_y = points.shape[1]
    image_x = points.shape[0]

    calib_path = path + 'calib/{}.txt'.format(img_id)
    calib = Calib(calib_path)

    velo_path = path + 'velodyne/{}.bin'.format(img_id)
    scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    velo_rect = calib.project_velo_to_rect(scan[:, :3])
    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(scan[:, :3], calib, 0, 0, image_x, image_y, True)
    label_file = path + 'label_2/{}.txt'.format(img_id)

=======

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def extract_frustum(path, points, img_id, index, point_dir, num_points, perturb_box2d=False, augmentX=1):
    image_y = points.shape[1]
    image_x = points.shape[0]

    calib_path = path + 'calib/{}.txt'.format(img_id)
    calib = Calib(calib_path)

    velo_path = path + 'velodyne/{}.bin'.format(img_id)
    scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    velo_rect = calib.project_velo_to_rect(scan[:, :3])
    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(scan[:, :3], calib, 0, 0, image_x, image_y, True)
    label_file = path + 'label_2/{}.txt'.format(img_id)
<<<<<<< HEAD

=======
>>>>>>> e8b96137bc6f28c671e46343aa6b0540a27e1356
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
    if os.path.isfile(label_file):
        objects = read_label(label_file)
    else:
        box_file = path + 'box/{}.txt'.format(img_id)
        objects = read_2d_box(box_file)
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276

    points = calib.project_image_to_rect(points.reshape(-1, 3)).reshape(image_y, image_x, 3)
    points_t = points.reshape(-1, 3)
    points_t = points_t[points_t[:, 2] != 0, :]
    choice = np.random.choice(points_t.shape[0], 5000, replace=True)
    points_t = points_t[choice]
    velo_rect_t = velo_rect[img_fov_inds, :]
    print(points_t.shape)
    print(velo_rect_t.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_t[:, 0].flatten(), points_t[:, 1].flatten(), points_t[:, 2].flatten(), c='r', marker='o')
    ax.scatter(velo_rect_t[:, 0].flatten(), velo_rect_t[:, 1].flatten(), velo_rect_t[:, 2].flatten(), c='b', marker='^')
    plt.show()
    return index, []

<<<<<<< HEAD
=======
=======
    image_x = points.shape[1]
    image_y = points.shape[0]
    points = project_image_to_rect(calib, points.reshape(-1, 3)).reshape(image_y, image_x, 3)
>>>>>>> e8b96137bc6f28c671e46343aa6b0540a27e1356
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
    for obj_idx in range(len(objects)):
        # 2D BOX: Get pts rect backprojected
        obj = objects[obj_idx]
        box2d = obj.box2d
        datas = {}
        box3d_count = []
        box3d_size = None
        if obj.type != 'Car' :
            continue

        for i in range(augmentX):
            # Augment data by box2d perturbation
            if perturb_box2d:
                xmin, ymin, xmax, ymax = random_shift_box2d(box2d).astype(int)
            else:
                xmin, ymin, xmax, ymax = box2d.astype(int)

            xmin = max(0, xmin)
            xmax = min(image_x-1, xmax)
            ymin = max(0, ymin)
            ymax = min(image_y-1, ymax)
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276

            if ymax - ymin < 25 or xmax < xmin :
                continue

            box_fov_inds = (pc_image_coord[:,0]<xmax) & (pc_image_coord[:,0]>=xmin) & (pc_image_coord[:,1]<ymax) & (pc_image_coord[:,1]>=ymin)

            box_fov_inds = box_fov_inds & img_fov_inds
            velo_in_box_fov = velo_rect[box_fov_inds,:]
            print(velo_in_box_fov.shape)

<<<<<<< HEAD
=======
            point_2d = points[ymin: ymax + 1, xmin: xmax + 1]
            point_2d = point_2d.reshape((-1, 3))


=======
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
            point_2d = points[ymin: ymax + 1, xmin: xmax + 1]
            point_2d = point_2d.reshape((-1, 3))
<<<<<<< HEAD


=======
            center_points = points[int(3 / 4.0 * ymin + ymax/4.0): int(3 / 4.0 * ymax + ymin/4.0) + 1, int(3 / 4.0 * xmin + xmax/4.0): int(3 / 4.0 * xmax + xmin/4.0) + 1]
            while center_points.size / 3 > 0.7 * num_points:
                print(center_points.shape)
                center_points = np.delete(center_points, list(range(0, center_points.shape[0], 8)), axis=0)
                center_points = np.delete(center_points, list(range(0, center_points.shape[1], 8)), axis=1)
            center_points = center_points.reshape((-1, 3))
            point_2d = np.vstack((point_2d, center_points))
>>>>>>> e8b96137bc6f28c671e46343aa6b0540a27e1356
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
            box2d_corner = np.array([xmin, ymin, xmax, ymax])
            box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0]).astype(int)

            center = points[(box2d_center[1], box2d_center[0])]
            frustum_angle = -1 * np.arctan2(center[2], center[0])
            if np.all(obj.t == 0):
                obj.t = center
            box3d_corner = compute_box_3d(obj)
<<<<<<< HEAD
            print(box3d_corner)
=======
<<<<<<< HEAD
            print(box3d_corner)
=======
>>>>>>> e8b96137bc6f28c671e46343aa6b0540a27e1356
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
            box3d_center = obj.t
            box3d_size = np.array([obj.l, obj.w, obj.h])
            _, inds = extract_pc_in_box3d(point_2d, box3d_corner)
            label = np.zeros(point_2d.shape[0])
            label[inds] = 1
            print(label.sum())
            if label.sum() == 0:
                print("skip")
                continue
            _, velo_inds = extract_pc_in_box3d(velo_in_box_fov, box3d_corner)
            velo_label = np.zeros(velo_in_box_fov.shape[0])
            velo_label[velo_inds] = 1
            print(velo_label.sum())
            if velo_label.sum() == 0:
                print("skip")
                continue
            datas['img_id'] = img_id
            datas['point_2d'] = point_2d
            datas['point_velo'] = velo_in_box_fov
            datas['box2d_corner'] = box2d_corner
            datas['box3d_corner'] = box3d_corner
            datas['box3d_center'] = box3d_center
            datas['box3d_size'] = box3d_size
            datas['frustum_angle'] = frustum_angle
            datas['heading'] = obj.ry
            datas['label'] = label
            datas['velo_label'] = velo_label
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
    path = 'obejct_data/data_object_image_2/training/'
    dir = path+'CNN_depth/'
    point_train = path+'frustum_points_train/'
    point_val = path+'frustum_points_val/'
    if not os.path.exists(point_train):
        os.makedirs(point_train)
    if not os.path.exists(point_val):
        os.makedirs(point_val)
    train_index = 0
    val_index = 0
    box3d_count = []
    for img in os.listdir(dir):
        img_id = img.split('.')[0]
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276

        points = find_3d_point(dir + img)
        if int(img_id) < 700:

<<<<<<< HEAD
=======
=======
        print(img_id)
        if int(img_id) < 600:
>>>>>>> e8b96137bc6f28c671e46343aa6b0540a27e1356
>>>>>>> f9ca4c90e1fc9695698c83af6390e3c514a18276
            train_index, box_count = extract_frustum(path,points, img_id, train_index, point_train, 1024, perturb_box2d=True,
                                                     augmentX=5)
            box3d_count = box3d_count + box_count
        else:
            val_index, box_count = extract_frustum(path,points, img_id, val_index, point_val, 1024, perturb_box2d=False,
                                                   augmentX=1)
            box3d_count = box3d_count + box_count
    np.savetxt('avg_box_size', np.mean(np.array(box3d_count), axis=0))
