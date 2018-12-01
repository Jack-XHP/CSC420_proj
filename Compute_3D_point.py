import numpy as np
import os
import cv2 as cv
import argparse
from KITTIloader2015 import read_label, Object3d, read_2d_box, Calib
from Plot_util import plot_all, plot_frsutum
from Box_util import random_shift_box2d, compute_box_3d, extract_pc_in_box3d


def find_3d_point(disp_path, calib):
    T = 0.54
    disp = cv.imread(disp_path, 0)
    depth = calib.f_u * T / disp + 9.0
    index = np.indices(disp.shape)
    x = index[1]
    y = index[0]
    points = np.vstack((x.flatten(), y.flatten(), depth.flatten())).T
    points = points.reshape((disp.shape[0], disp.shape[1], 3))
    return points


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def extract_frustum(path, img_id, index, point_dir, perturb_box2d=False, augmentX=1, demo=False):
    calib_path = path + 'calib/{}.txt'.format(img_id)
    calib = Calib(calib_path)

    disp_path = path + 'CNN_depth/{}.png'.format(img_id)
    points_raw = find_3d_point(disp_path, calib)

    image_x = points_raw.shape[1]
    image_y = points_raw.shape[0]

    points = calib.project_image_to_rect(points_raw.reshape(-1, 3)).reshape(image_y, image_x, 3)

    velo_path = path + 'velodyne/{}.bin'.format(img_id)
    scan = np.fromfile(velo_path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    velo_rect = calib.project_velo_to_rect(scan[:, :3])
    _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(scan[:, :3], calib, 0, 0, image_x, image_y, True)

    label_file = path + 'label_2/{}.txt'.format(img_id)
    objects = read_label(label_file)

    for obj_idx in range(len(objects)):
        # 2D BOX: Get pts rect backprojected
        obj = objects[obj_idx]
        box2d = obj.box2d
        box3d_count = []
        box3d_size = None
        if obj.type != 'Car':
            continue

        for i in range(augmentX):
            # Augment data by box2d perturbation
            if perturb_box2d:
                xmin, ymin, xmax, ymax = random_shift_box2d(box2d).astype(int)
            else:
                xmin, ymin, xmax, ymax = box2d.astype(int)

            xmin = max(0, xmin)
            xmax = min(image_x - 1, xmax)
            ymin = max(0, ymin)
            ymax = min(image_y - 1, ymax)

            if ymax - ymin < 25 or xmax < xmin:
                continue
            print("2d box {}, {}, {}, {}".format(xmin, ymin, xmax, ymax))

            box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & (
                    pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)

            box_fov_inds = box_fov_inds & img_fov_inds
            velo_in_box_fov = velo_rect[box_fov_inds, :]

            point_2d = points_raw[ymin: ymax + 1, xmin: xmax + 1]
            point_2d = point_2d.reshape((-1, 3))
            mask = point_2d[:, 2] < 100
            point_2d = point_2d[mask, :]
            depth_shift = np.mean(velo_in_box_fov[:, 2]) - np.mean(point_2d[:, 2])
            print("depth shift = {}".format(depth_shift))
            point_2d[:, 2] += depth_shift
            point_2d = calib.project_image_to_rect(point_2d)
            center_shift = np.mean(velo_in_box_fov, axis=0) - np.mean(point_2d, axis=0)
            print("center shift = {}".format(center_shift))
            point_2d += center_shift

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
            label = np.zeros(point_2d.shape[0])
            label[inds] = 1

            print("rgb points in box {}".format(label.sum()))
            if label.sum() == 0:
                print("skip")
                continue

            _, velo_inds = extract_pc_in_box3d(velo_in_box_fov, box3d_corner)
            velo_label = np.zeros(velo_in_box_fov.shape[0])
            velo_label[velo_inds] = 1
            print("lidar points in box {}".format(velo_label.sum()))
            if velo_label.sum() == 0:
                print("skip")
                continue

            if demo:
                plot_frsutum(box3d_corner, point_2d, velo_in_box_fov)
            else:
                datas = {}
                datas['img_id'] = img_id
                datas['point_2d'] = point_2d
                datas['point_velo'] = velo_in_box_fov
                datas['box2d_corner'] = box2d_corner
                datas['box3d_center'] = box3d_center
                datas['box3d_size'] = box3d_size
                datas['frustum_angle'] = frustum_angle
                datas['heading'] = obj.ry
                datas['label'] = label
                datas['velo_label'] = velo_label
                np.save(point_dir + str(index), datas)
                index += 1

        if box3d_size is not None:
            box3d_count.append(box3d_size)

    if demo:
        plot_all(points, image_y, image_x, calib, velo_rect[img_fov_inds, :])

    return index, box3d_count


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comput 3d points')
    parser.add_argument('--datapath', default='obejct_data/data_object_image_2/training/', help='datapath')
    parser.add_argument('--demo', default=False, help='load model')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    args = parser.parse_args()
    np.random.seed(args.seed)
    if args.demo:
        args.demo = True
    path = args.datapath
    dir = path + 'CNN_depth/'
    point_train = path + 'frustum_points_train/'
    point_val = path + 'frustum_points_val/'
    if not os.path.exists(point_train):
        os.makedirs(point_train)
    if not os.path.exists(point_val):
        os.makedirs(point_val)
    train_index = 0
    val_index = 0
    box3d_count = []
    for img in os.listdir(dir):
        img_id = img.split('.')[0]
        if int(img_id) < 5500:
            train_index, box_count = extract_frustum(path, img_id, train_index, point_train, perturb_box2d=True,
                                                     augmentX=2, demo=args.demo)
            box3d_count = box3d_count + box_count
        else:
            val_index, box_count = extract_frustum(path, img_id, val_index, point_val, perturb_box2d=False,
                                                   augmentX=1, demo=args.demo)
            box3d_count = box3d_count + box_count
    np.savetxt('avg_box_size', np.mean(np.array(box3d_count), axis=0))
