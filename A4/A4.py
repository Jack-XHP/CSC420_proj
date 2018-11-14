import numpy as np
import cv2 as cv
import scipy
import matplotlib.pyplot as plt
import os


def computeDepth():
    # get the image list and loop all the images
    imageList = open('data/test/test.txt')
    images = imageList.readlines()
    i = 0
    for image in images:
        image = image.rstrip()
        # get calibration file
        calFile = open('data/test/calib/{}_allcalib.txt'.format(image))
        cal = calFile.readlines()
        cal = [float(p.split()[1]) for p in cal]
        disparity = cv.imread('data/test/results/{}_left_disparity.png'.format(image), 0)
        # depth = f * baseline / disparity
        depth = cal[0] * cal[3] / disparity
        if i < 3:
            # show disparity and depth
            plt.matshow(depth)
            plt.show()
        i += 1
        # save depth info in result folder
        np.savetxt('data/test/results/{}_depth.txt'.format(image), depth)


def visualBox():
    # get the image list and loop all the images
    imageList = open('data/test/test.txt')
    images = imageList.readlines()
    colors = {1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0), 10: (0, 255, 255)}
    className = {1: 'person', 2: 'bicycle', 3: 'car', 10: 'traffic light'}
    for i in range(3):
        image = images[i].rstrip()
        im = cv.imread('data/test/left/{}.jpg'.format(image))
        im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        # get box and classes result of object detector
        boxes = np.loadtxt('data/test/results/{}_box.txt'.format(image)) * (im.shape[:2] + im.shape[:2])
        boxes = boxes.astype(int)
        classes = np.loadtxt('data/test/results/{}_class.txt'.format(image))
        for i in range(classes.shape[0]):
            # a box for each object detected
            cv.rectangle(im, (boxes[i][1], boxes[i][0]), (boxes[i][3], boxes[i][2]), colors[classes[i]], 7)
            # put the class name of the object
            name = className[classes[i]]
            cv.putText(im, name, (boxes[i][3] - 16 * len(name), boxes[i][0] - 10), cv.FONT_HERSHEY_SIMPLEX, 1.2,
                       colors[classes[i]], 5)
        plt.imshow(im)
        plt.show()


def center3D():
    # get the image list and loop all the images
    imageList = open('data/test/test.txt')
    images = imageList.readlines()
    for image in images:
        image = image.rstrip()
        im = cv.imread('data/test/left/{}.jpg'.format(image))
        # get box object detector
        boxes = np.loadtxt('data/test/results/{}_box.txt'.format(image)) * (im.shape[:2] + im.shape[:2])
        boxes = boxes.astype(int)
        # get depth info
        depth = np.loadtxt('data/test/results/{}_depth.txt'.format(image))
        # get calibration info
        calFile = open('data/test/calib/{}_allcalib.txt'.format(image))
        cal = calFile.readlines()
        cal = [float(p.split()[1]) for p in cal]
        center_depth = []
        for i in range(boxes.shape[0]):
            # for each detected object, get the depth info in its box
            box = boxes[i]
            box_depth = depth[box[0]: box[2] + 1, box[1]:box[3] + 1]
            box_depth[np.isinf(box_depth)] = 0
            # get histogram of all pixels depth within that box
            hist, bin = np.histogram(box_depth.flatten(), 'auto')
            # use the most common depth as center of mass's depth
            max_bin = np.argmax(hist)
            box_depth = (bin[max_bin] + bin[max_bin + 1]) / 2.0
            # use depth and box center to get 3D center of mass
            center = reprojectTo3D(cal, (box[1] + box[3]) / 2.0, (box[0] + box[2]) / 2.0, box_depth)
            center_depth.append(center)
        # save the boxes' center of mass to result folder
        center_depth = np.array(center_depth)
        np.savetxt('data/test/results/{}_3D.txt'.format(image), center_depth)


def reprojectTo3D(cal, x, y, z):
    # as K * [X, Y, Z].T = w[x, y, 1]
    # x = f*X/Z + px
    # y = f*Y/Z + py
    X = (x - cal[1]) * z / cal[0]
    Y = (y - cal[2]) * z / cal[0]
    return [X, Y, z]


def distance3D(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))


def segment():
    # get the image list and loop all the images
    imageList = open('data/test/test.txt')
    images = imageList.readlines()
    for i in range(3):
        image = images[i].rstrip()
        im = cv.imread('data/test/left/{}.jpg'.format(image))
        # get boxes result
        boxes = np.loadtxt('data/test/results/{}_box.txt'.format(image)) * (im.shape[:2] + im.shape[:2])
        boxes = boxes.astype(int)
        # get depth info
        depth = np.loadtxt('data/test/results/{}_depth.txt'.format(image))
        # get center of mass 3D point info
        center = np.loadtxt('data/test/results/{}_3D.txt'.format(image))
        # get calibration info
        calFile = open('data/test/calib/{}_allcalib.txt'.format(image))
        cal = calFile.readlines()
        cal = [float(p.split()[1]) for p in cal]
        seg = np.zeros((im.shape[0], im.shape[1]))
        for i in range(boxes.shape[0]):
            box = boxes[i]
            for y in range(box[0], box[2]):
                for x in range(box[1], box[3]):
                    # for each pixel within a box, get it 3D point using picture coordinate and depth
                    point = reprojectTo3D(cal, x, y, depth[y][x])
                    # compute its distance from center of mass
                    distance = distance3D(point, center[i])
                    if distance <= 3:
                        seg[y][x] = i+1
        plt.matshow(seg, cmap='coolwarm')
        plt.show()


def sentence():
    # get the image list and loop all the images
    imageList = open('data/test/test.txt')
    images = imageList.readlines()
    className = {1: 'person', 2: 'bicycle', 3: 'car', 10: 'traffic light'}
    for image in images:
        image = image.rstrip()
        # get center of mass 3D point info
        center = np.loadtxt('data/test/results/{}_3D.txt'.format(image))
        # get class results
        classes = np.loadtxt('data/test/results/{}_class.txt'.format(image))
        # count each class' appearance
        people = np.count_nonzero(classes == 1)
        bicy = np.count_nonzero(classes == 2)
        car = np.count_nonzero(classes == 3)
        light = np.count_nonzero(classes == 10)
        summary = "There are {} people, {} bicycles, {} cars and {} traffic light in the scene".format(people, bicy,
                                                                                                       car, light)
        # compute distance of each box to camera origin
        ori = np.zeros(3)
        dist = np.array([distance3D(ori, point) for point in center])
        # find closest box
        closest = np.argmin(dist)
        # determine its class and location
        name = className[classes[closest]]
        direction = 'left' if center[closest][0] < 0 else 'right'
        closest = "The closest {} is {} meters away to your {}".format(name, dist[closest], direction)
        # print both sentences
        print(summary)
        print(closest)


computeDepth()
visualBox()
center3D()
segment()
sentence()
