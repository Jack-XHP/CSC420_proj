"""
adapted from https://github.com/JiaRenChang/PSMNet for
"Pyramid Stereo Matching Network" paper (CVPR 2018) by Jia-Ren Chang and Yong-Sheng Chen.
"""
import torch.utils.data as data
import os
import os.path

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
