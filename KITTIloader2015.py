"""
adapted from https://github.com/JiaRenChang/PSMNet for
"Pyramid Stereo Matching Network" paper (CVPR 2018) by Jia-Ren Chang and Yong-Sheng Chen.
"""
import torch.utils.data as data
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, load):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp_L = 'disp_occ_0/'

  image = [img for img in os.listdir(filepath+left_fold)]

  if load is not None:
    train = []
    val = image
    disp_train_L = []
    disp_val_L = []
  else:
    train = image[:160]
    val   = image[160:]
    disp_train_L = [filepath+disp_L+img for img in train]
    disp_val_L = [filepath+disp_L+img for img in val]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]


  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]


  return left_train, right_train, disp_train_L, left_val, right_val, disp_val_L, train, val


def obeject_dataloader(filepath):

  left_fold  = 'image_2/'
  right_fold = 'image_3/'
  disp = 'CNN_depth/'
  label = 'label_2/'

  image = [img for img in os.listdir(filepath+left_fold)]

  train = image[:700]
  val   = image[700:]
  disp_train = [filepath+disp+img for img in train]
  disp_val = [filepath+disp+img for img in val]
  label_train = [filepath+label+img.split('.')[0]+'.txt' for img in train]
  label_val = [filepath + label + img.split('.')[0] + '.txt' for img in val]

  left_train  = [filepath+left_fold+img for img in train]
  right_train = [filepath+right_fold+img for img in train]


  left_val  = [filepath+left_fold+img for img in val]
  right_val = [filepath+right_fold+img for img in val]


  return left_train, right_train, disp_train, left_val, right_val, disp_val, train, val, label_train, label_val


class Object3d(object):
  ''' 3d object label '''

  def __init__(self, label_file_line):
    data = label_file_line.split(' ')
    data[1:] = [float(x) for x in data[1:]]

    # extract label, truncation, occlusion
    self.type = data[0]  # 'Car', 'Pedestrian', ...
    self.truncation = data[1]  # truncated pixel ratio [0..1]
    self.occlusion = int(data[2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
    self.alpha = data[3]  # object observation angle [-pi..pi]

    # extract 2d bounding box in 0-based coordinates
    self.xmin = data[4]  # left
    self.ymin = data[5]  # top
    self.xmax = data[6]  # right
    self.ymax = data[7]  # bottom
    self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

    # extract 3d bounding box information
    self.h = data[8]  # box height
    self.w = data[9]  # box width
    self.l = data[10]  # box length (in meters)
    self.t = np.array([data[11], data[12], data[13]]) # location (x,y,z) in camera coord.
    self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

  def print_object(self):
    print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
          (self.type, self.truncation, self.occlusion, self.alpha))
    print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
          (self.xmin, self.ymin, self.xmax, self.ymax))
    print('3d bbox h,w,l: %f, %f, %f' % \
          (self.h, self.w, self.l))
    print('3d bbox location, ry: (%f, %f, %f), %f' % \
          (self.t[0], self.t[1], self.t[2], self.ry))

def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects

def read_2d_box(box_file):
    objects = []
    empty_list = "Car 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
    for line in open(box_file):
        data = line.split('')
        new_obj = Object3d(empty_list)
        new_obj.type = 'Car'
        new_obj.xmin = data[0]
        new_obj.ymin = data[1]
        new_obj.xmax = data[2]
        new_obj.ymax = data[3]
        new_obj.box2d = np.array(data)
        objects.append(new_obj)
    return objects
