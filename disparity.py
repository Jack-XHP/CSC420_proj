import matplotlib.pyplot as plt
import cv2
import numpy as np
from numpy.linalg import inv
from scipy.linalg import *
# from sympy import Matrix
import math

def detect(img,template,pat):
	res = cv2.matchTemplate(img,template,cv2.TM_CCORR_NORMED)
	min_val , max_val , min_loc , max_loc = cv2.minMaxLoc(res)
	top_left = max_loc
	xr = top_left[0]+pat
	return xr

def read_box_2d(label_file):
	boxes = []
	for line in open(label_file):
		data = line.strip().split(' ')
		if data[0]!='Car':
			continue
		xmin = int(float(data[4]))
		ymin = int(float(data[5]))
		xmax = int(float(data[6]))
		ymax = int(float(data[7]))
		box2d = np.array([xmin,ymin,xmax,ymax])
		boxes.append(box2d)
	return boxes

def compute_disparity(left_img, right_img, PAT, xmin):
	disparity = np.zeros(left_img.shape)
	for row in range(0,left_img.shape[0]):
		for col in range(0,left_img.shape[1]):
			pl = left_img[max(0,row-PAT):min(left_img.shape[0]-1,row+PAT),max(0,col-PAT):min(left_img.shape[1]-1,col+PAT)]
			xl = col+xmin
			xr = detect(right_img[max(0,row-PAT):min(left_img.shape[0]-1,row+PAT),:,:],pl,PAT)
			disparity[row][col] = xl-xr

	plt.imshow(disparity,cmap='gray')
	plt.show()
	return disparity

def main():
	box_path = "./obejct_data/data_object_image_2/training/label_2/"
	left_path = "./obejct_data/data_object_image_2/training/image_2/"
	right_path = "./obejct_data/data_object_image_2/training/image_3/"
	PAT = 5
	for i in range(8,9):
		print("image_id:" + str(i))
		labelfile = box_path + str(i).zfill(6)+".txt"
		left_img_file = left_path+str(i).zfill(6)+".png"
		left_img = cv2.imread(left_img_file)
		right_img_file = right_path+str(i).zfill(6)+".png"
		right_img = cv2.imread(right_img_file)
		boxes = read_box_2d(labelfile)
		whole_img_dis = compute_disparity(left_img,right_img,PAT,0)
		for box in boxes:
			xmin, ymin, xmax, ymax = box
			print(xmin,ymin,xmax,ymax)
			left_box = left_img[ymin:ymax+1,xmin:xmax+1,:]
			right_line = right_img[ymin:ymax+1,:,:]
			disparity = compute_disparity(left_box,right_line, PAT,xmin)		

if __name__ == '__main__':
	main()

