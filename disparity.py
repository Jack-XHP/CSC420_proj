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
if __name__ == '__main__':
	left_path = "./obejct_data/data_object_image_2/training/image_2/"
	right_path = "./obejct_data/data_object_image_2/training/image_3/"
	PAT = 5
	for i in range(3,6):
		print(i)
		left_img_file = left_path+str(i).zfill(6)+".png"
		left_img = cv2.imread(left_img_file)
		left_img = cv2.cvtColor(left_img , cv2.COLOR_BGR2GRAY)
		right_img_file = right_path+str(i).zfill(6)+".png"
		right_img = cv2.imread(right_img_file)
		right_img = cv2.cvtColor(right_img , cv2.COLOR_BGR2GRAY)
		disparity = np.zeros(left_img.shape)
		for row in range(0,left_img.shape[0]):
			for col in range(0,left_img.shape[1]):
				pl = left_img[max(0,row-PAT):min(left_img.shape[0]-1,row+PAT),max(0,col-PAT):min(left_img.shape[1]-1,col+PAT)]
				xl = col
				xr = detect(right_img[max(0,row-PAT):min(left_img.shape[0]-1,row+PAT),:],pl,PAT)
				disparity[row][col] = xl-xr
		print(disparity)
		plt.imshow(disparity,cmap='gray')
		plt.show()

