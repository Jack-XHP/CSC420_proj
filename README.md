# CSC420_proj

Obejct Data Format Description
=======================

The data for training and testing can be found in the corresponding folders.
The sub-folders are structured as follows:

  - image_02/ contains the left color camera images (png)
  - label_02/ contains the left color camera label files (plain text files)
  - calib/ contains the calibration for all four cameras (plain text file)

The label files contain the following information, which can be read and
written using the matlab tools (readLabels.m, writeLabels.m) provided within
this devkit. All values (numerical or strings) are separated via spaces,
each row corresponds to one object. The 15 columns represent:

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.

The camera images are stored in the following directories:

  - 'image_00': left rectified grayscale image sequence
  - 'image_01': right rectified grayscale image sequence
  - 'image_02': left rectified color image sequence
  - 'image_03': right rectified color image sequence

We are using cemera 02 and 03 (color left and right)
calib_cam_to_cam.txt: Camera-to-camera calibration
--------------------------------------------------

  - S_xx: 1x2 size of image xx before rectification
  - K_xx: 3x3 calibration matrix of camera xx before rectification
  - D_xx: 1x5 distortion vector of camera xx before rectification
  - R_xx: 3x3 rotation matrix of camera xx (extrinsic)
  - T_xx: 3x1 translation vector of camera xx (extrinsic)
  - S_rect_xx: 1x2 size of image xx after rectification
  - R_rect_xx: 3x3 rectifying rotation to make image planes co-planar
  - P_rect_xx: 3x4 projection matrix after rectification
