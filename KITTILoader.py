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

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

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


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, name, load=False, loader=default_loader, dploader=disparity_loader):

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
