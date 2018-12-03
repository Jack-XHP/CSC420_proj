import os
import sys
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
from matplotlib import pyplot as plt
from data import VOC_CLASSES as labels


def object_detect(img, out_box_dir, draw_box, out_img_dir):

    image = cv2.imread(img, cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # View the sampled input image before transform
    # plt.figure(figsize=(10,10))
    # plt.imshow(rgb_image)
    # plt.show()


    x = cv2.resize(image, (300, 300)).astype(np.float32)
    x -= (104.0, 117.0, 123.0)
    x = x.astype(np.float32)
    x = x[:, :, ::-1].copy()
    # plt.imshow(x)
    # plt.show()
    x = torch.from_numpy(x).permute(2, 0, 1)


    xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
    # if torch.cuda.is_available():
    #     xx = xx.cuda()
    y = net(xx)



    # top_k=10
    # plt.figure()
    # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # plt.imshow(rgb_image)  # plot the image for matplotlib
    # currentAxis = plt.gca()
    out_box = ""
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.1:
            score = detections[0,i,j,0]
            label_name = labels[i-1]
            display_txt = '%s: %.2f'%(label_name, score)
            pt = (detections[0,i,j,1:]*scale).cpu().numpy()
            # print(pt)
            if label_name == "car":
                out_box = out_box + str(pt) + '\n'

            # coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
            # color = colors[i]
            color = (255, 0, 0)
            cv2.rectangle(rgb_image, (pt[0], pt[1]), (pt[2], pt[3]), color, 1)
            t_size = cv2.getTextSize(label_name, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
            cv2.putText(rgb_image, label_name, (int(pt[0]), int(pt[1]) + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)

            # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
            # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
            j+=1

    filename = img.split("/")[-1].split(".")[-2]

    if draw_box:
        # print(filename)
        draw_path = out_img_dir + '/det_{}.jpg'.format(filename)
        cv2.imwrite(draw_path, rgb_image)

    out_path = out_box_dir + '/{}.txt'.format(filename)
    with open(out_path, 'w') as f:
        out = str(out_box)
        out = out.replace('[', "")
        out = out.replace(']', "")
        f.write(out)
    # plt.show()
    # plt.savefig('../det/{}_out.jpg'.format(img.split("/")[-1].split(".")[-2]))

def arg_parse():
    """
    Parse arguements to the detect module

    """


    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_mutually_exclusive_group(required=False)

    parser.add_argument('--datapath', dest = 'datapath', default='obejct_data/data_object_image_2/training/', help='datapath')
    parser.add_argument("--images", dest = 'images', help =
                        "Image / Directory containing images to perform detection upon",
                        default = 'obejct_data/data_object_image_2/training/image_2/', type = str)
    parser.add_argument("--det", dest = 'det', help =
                        "Image / Directory to store detections to",
                        default = "det", type = str)
    parser.add_argument('--draw-box', dest = 'draw_box', action='store_true')
    parser.add_argument('--no-draw-box', dest = 'draw_box', action='store_false')
    parser.set_defaults(draw_box=False)
    parser.add_argument("--weights", dest = 'weightsfile', help =
                        "weightsfile",
                        default = "ssd300_mAP_77.43_v2.pth", type = str)

    return parser.parse_args()

if __name__ ==  '__main__':
    args = arg_parse()

    net = build_ssd('test', 300, 21)    # initialize SSD
    net.load_weights(args.weightsfile)

    parent_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/"
    imageList = parent_folder + args.images
    # print(imageList)
    out_box_dir = parent_folder + args.datapath + 'box_ssd'

    # out_box_dir = os.path.dirname(os.path.realpath(__file__)) + '/out_box'
    if not os.path.exists(out_box_dir):
        os.makedirs(out_box_dir)

    out_img_dir = os.path.dirname(os.path.realpath(__file__))+'/'+ args.det
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)

    # imageList = '../obejct_data/data_object_image_2/training/image_2/'
    # imageList = './images/'
    TEST_IMAGE_PATHS = [imageList+img for img in os.listdir(imageList)]
    for img_path in TEST_IMAGE_PATHS:
        object_detect(img_path, out_box_dir, args.draw_box, out_img_dir)
