import numpy as np
import os
import cv2 as cv


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

    def cloud_density(self, box_center, orientation, size):



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
    mask = depth.flatten() < 1000
    points = points[mask, ...]
    return points


if __name__ == '__main__':
    dir = 'obejct_data/data_object_image_2/training/CNN_depth/'
    for img in os.listdir(dir):
        points = find_3d_point(dir + img)
        voxel = Voxel(0.2)
        voxel.discrete_points(points)
