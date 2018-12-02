import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s],
    				[0, 1, 0],
    				[-s, 0, c]])

def project_to_image(pts_3d, P):
	# ''' Project 3d points to image plane.
	# Usage: pts_2d = projectToImage(pts_3d, P)
	# input: pts_3d: nx3 matrix
	# P:      3x4 projection matrix
	# output: pts_2d: nx2 matrix
	n = pts_3d.shape[0]
	pts_3d_extend = np.hstack((pts_3d, np.ones((n,1))))
	pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # nx3
	pts_2d[:,0] /= pts_2d[:,2]
	pts_2d[:,1] /= pts_2d[:,2]
	return pts_2d[:,0:2]

def read_calib_file(filepath):
	data = {}
	with open(filepath, 'r') as f:
		for line in f.readlines():
			line = line.rstrip()
			if len(line)==0: continue
			key, value = line.split(':', 1)
			try:
				data[key] = np.array([float(x) for x in value.split()])
			except ValueError:
				pass
	return data

def draw_projected_box3d(image, qs, color=(0,0,255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
    return image

def saveBGR2RGB( im, savename):
	img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	plt.imsave(savename,img)
	return

def read_file_name(box3dcornerforlder):
	result = {}
	for name in os.listdir(box3dcornerforlder):
		n = name[:-4]
		n = n.split('_')
		if n[0] in result:
			result[n[0]].append(n[1])
		else:
			result[n[0]] = [n[1]]
	return result

def main():
	imgforlder = "./obejct_data/data_object_image_2/training/image_2/"
	calibforder = "./obejct_data/data_object_image_2/training/calib/"
	box3dcornerforlder = "./obejct_data/data_object_image_2/training/demo_result/"
	save_folder = "./obejct_data/data_object_image_2/training/result_image/"
	dct = read_file_name(box3dcornerforlder)
	imgsavefolder = ""
	for imgid in dct.keys():
		# print(imgid)
		imgpath = imgforlder + imgid+".png"
		img = cv2.imread(imgpath)
		calib_path = calibforder + imgid + ".txt"
		calibs = read_calib_file(calib_path)
		for boxid in dct[imgid]:
			box3dcornerpath = box3dcornerforlder + imgid + "_" + boxid + ".npy"
			P = calibs['P2'] 
			P = np.reshape(P, [3,4])
			points = np.load(box3dcornerpath)
			frustum_angle = points[-1]
			points = points[:-1]
			points = np.array(points).reshape(8,3)  #8x3
			# print(points)
			yrotation = roty(frustum_angle)  #3x3
			rotatepoints = np.matmul(yrotation,np.transpose(points)) #3x8
			points2d = project_to_image(np.transpose(rotatepoints),P)  #8x2
			# print(frustum_angle)
			# print(points2d)
			img = draw_projected_box3d(img,points2d)
		saveBGR2RGB(img,save_folder+imgid+".png")

if __name__ == '__main__':
	main()