"""
adapted from https://github.com/JiaRenChang/PSMNet for
"Pyramid Stereo Matching Network" paper (CVPR 2018) by Jia-Ren Chang and Yong-Sheng Chen.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2 as cv
import time
import math
import KITTIloader2015 as ls
import KITTILoader as DA
import os


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)), [1, maxdisp, 1, 1])),
                             requires_grad=False)
        if args.cuda:
            self.disp =self.disp.cuda()

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
        out = torch.sum(x * disp, 1)
        return out


def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False),
        nn.BatchNorm3d(out_planes))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64, 64)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32, 32)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16, 16)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8, 8)),
                                     convbn(128, 32, 1, 1, 0, 1),
                                     nn.ReLU(inplace=True))

        self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128, 32, kernel_size=1, padding=0, stride=1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.firstconv(x)
        output = self.layer1(output)
        output_raw = self.layer2(output)
        output = self.layer3(output_raw)
        output_skip = self.layer4(output)

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.upsample(output_branch1, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.upsample(output_branch2, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.upsample(output_branch3, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.upsample(output_branch4, (output_skip.size()[2], output_skip.size()[3]), mode='bilinear')

        output_feature = torch.cat(
            (output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        output_feature = self.lastconv(output_feature)

        return output_feature


class PSMNet(nn.Module):
    def __init__(self, maxdisp):
        super(PSMNet, self).__init__()
        self.maxdisp = maxdisp
        self.feature_extraction = feature_extraction()

        ########
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.dres4 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.classify = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, left, right):

        refimg_fea = self.feature_extraction(left)
        targetimg_fea = self.feature_extraction(right)

        # matching
        cost = Variable(
            torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1] * 2, self.maxdisp / 4, refimg_fea.size()[2],
                              refimg_fea.size()[3]).zero_(), volatile=not self.training)
        if args.cuda:
            cost = cost.cuda()

        for i in range(self.maxdisp / 4):
            if i > 0:
                cost[:, :refimg_fea.size()[1], i, :, i:] = refimg_fea[:, :, :, i:]
                cost[:, refimg_fea.size()[1]:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                cost[:, :refimg_fea.size()[1], i, :, :] = refimg_fea
                cost[:, refimg_fea.size()[1]:, i, :, :] = targetimg_fea
        cost = cost.contiguous()

        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0
        cost0 = self.dres2(cost0) + cost0
        cost0 = self.dres3(cost0) + cost0
        cost0 = self.dres4(cost0) + cost0

        cost = self.classify(cost0)
        cost = F.upsample(cost, [self.maxdisp, left.size()[2], left.size()[3]], mode='trilinear')
        cost = torch.squeeze(cost, 1)
        pred = F.softmax(cost)
        pred = disparityregression(self.maxdisp)(pred)

        return pred


def train(imgL, imgR, disp_true):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_true = Variable(torch.FloatTensor(disp_true))
    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()
    output = model(imgL, imgR)
    output = torch.squeeze(output, 1)
    loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data[0]


def test(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.data.cpu()

    # computing 3-px error#
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
            disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
        index[0][:], index[1][:], index[2][:]] * 0.05)
    if args.cuda:
        torch.cuda.empty_cache()
    else:
        torch.empty_cache()

    return 1 - (float(torch.sum(correct)) / float(len(index[0])))


def result(imgL, imgR, disp_true, name):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.data.cpu().numpy().astype(np.uint16)
    print(args.datapath+'CNN_depth/'+name[0])
    cv.imwrite(args.datapath+'CNN_depth/'+name[0], pred_disp[0])

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = 0.001
    else:
        lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    max_acc = 0
    max_epo = 0
    start_full_time = time.time()

    for epoch in range(1, args.epochs + 1):
        total_train_loss = 0
        total_test_loss = 0
        adjust_learning_rate(optimizer, epoch)

        ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L, name) in enumerate(TrainImgLoader):
            start_time = time.time()

            loss = train(imgL_crop, imgR_crop, disp_crop_L)
            print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, loss, time.time() - start_time))
            total_train_loss += loss
        print('epoch %d total training loss = %.3f' % (epoch, total_train_loss / len(TrainImgLoader)))

        ## Test ##

        for batch_idx, (imgL, imgR, disp_L, name) in enumerate(TestImgLoader):
            test_loss = test(imgL, imgR, disp_L)
            print('Iter %d 3-px error in val = %.3f' % (batch_idx, test_loss * 100))
            total_test_loss += test_loss

        print('epoch %d total 3-px error in val = %.3f' % (epoch, total_test_loss / len(TestImgLoader) * 100))
        if total_test_loss / len(TestImgLoader) * 100 > max_acc:
            max_acc = total_test_loss / len(TestImgLoader) * 100
            max_epo = epoch
        print('MAX epoch %d total test error = %.3f' % (max_epo, max_acc))

        # SAVE
        savefilename = args.savemodel + 'finetune_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
            'test_loss': total_test_loss / len(TestImgLoader) * 100,
        }, savefilename)

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time) / 3600))
    print(max_epo)
    print(max_acc)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN_disp')
    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--datapath', default='depth_data/data/training/',
                        help='datapath')
    parser.add_argument('--epochs', type=int, default=300,
                        help='number of epochs to train')
    parser.add_argument('--savemodel', default='./',
                        help='save model')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--loadmodel', default=None,
                        help='load model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp, train_name, val_name = ls.dataloader(
        args.datapath, args.loadmodel)

    if args.loadmodel is not None:
        ResultImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, val_name, load=True),
            batch_size=1, shuffle=False, num_workers=8, drop_last=False)
    else:
        TrainImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(all_left_img, all_right_img, all_left_disp, True, train_name),
            batch_size=4, shuffle=True, num_workers=8, drop_last=False)

        TestImgLoader = torch.utils.data.DataLoader(
            DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False, val_name),
            batch_size=4, shuffle=False, num_workers=8, drop_last=False)

    model = PSMNet(args.maxdisp)
    if args.loadmodel is not None:
        if args.cuda:
            state_dict = torch.load(args.loadmodel)
        else:
            state_dict = torch.load(args.loadmodel, map_location='cpu')
        model.load_state_dict(state_dict['state_dict'])
    if args.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))

    if args.loadmodel is not None:
        directory = args.datapath+'CNN_depth/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for batch_idx, (imgL, imgR, disp_L, name) in enumerate(ResultImgLoader):
            result(imgL, imgR, disp_L, name)
    else:
        main()
