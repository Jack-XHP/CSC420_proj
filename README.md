# CSC420_proj

Require packages:
===========
- python 2.7
- pytorch 0.4.1
- tensorflow 1.12.0
- CUDA 9.2
- numpy
- scipy
- opencv3.4.3
- for detailed list of packages check requirement.txt

Data folder structure
========
- To train PSMnet in CNN_disparity.py, download KITTI stereo 2015 task http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
    - form a data folder like this:
       - depth_data:
            - disp_occ_0
            - image_2
            - image_3

- To train Point net in Point2Box.py, download KITTI 3D object detection task http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d
    - form a data folder like this:
        - object_data:
            - image_2
            - image_3
            - calib
            - label_2
            - velodyne
            - CNN_depth (get from CNN_disparity.py)
            - frsutum_points_train ( get from Compute_3D_point.py)
            - frsutum_points_val ( get from Compute_3D_point.py)
            - frsutum_points_test ( get from Compute_3D_point.py)
            - box_ssd (get from part1ssd)
            - box_yolo (get from part1yolo)
- Get Trained models for PSMnet and Point net from https://github.com/Jack-XHP/CSC420_proj/tree/master/trained_models

current scripts for part1:
==========
- get 2D box (yolov3):  download yolov3.weights from here https://pjreddie.com/media/files/yolov3.weights

    Put the weights file into part1yolo folder
    ```
    cd part1yolo
    pythonw detect.py
    ```

    IF you want to draw the 2D box on images, add argument `--draw-box`, output will be in /det folder
    Thanks for the tutorial and code from https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/

- get 2D box (ssd): download ssd weights (ssd300_mAP_77.43_v2.pth) from here https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth

    Put the weights file into part1ssd folder
    ```
    cd part1ssd
    pythonw detect.py
    ```
    IF you want to draw the 2D box on images, add argument `--draw-box`,  output will be in /det folder
    Thanks for code reference from https://github.com/amdegroot/ssd.pytorch

current scripts:
==========
- get Point Net prepared data:
    ```
    python Compute_3D_point.py --datapath [data folder] --demo [T/F] --trainsize [training set image range] --nolabel [T/F]
    ```
    - get training and validation set with ground truth label: 
        ```
        python Compute_3D_point.py --datapath [data folder] --trainsize [training set image range]
        ```
    - get test set with CNN estimated 2D box: 
        ```
        python Compute_3D_point.py --datapath [data folder] --nolabel True
        ```
    - get plot of 3D points for each image and each frustum of 2D box: 
        ```
        python Compute_3D_point.py --datapath [data folder] --demo True
        ```
- PSMnet source code : https://github.com/JiaRenChang/PSMNet/
    - train model: 
        ```
        python CNN_disparity.py  --datapath [data folder]
        ```
    - get CNN disparity:  
        ```
        python CNN_disparity.py --loadmodel [model] --datapath [data folder]
        ```
- Point Net source code : https://github.com/charlesq34/frustum-pointnets
    - Point nets: 
        ```
        Point2Box.py [-h] --datapath [DATAPATH] --loadmodel [LOADMODEL] --uselidar [T/F]
        ```
    - train Point net with disparity points: 
        ```
        python Point2Box.py --datapath [data folder]
        ```
    - train Point net with Lidar points: 
        ```
        python Point2Box.py --datapath [data folder] --uselidar True
        ```
    - get Point net estimate 3D box using disparity points: 
        ```
        python Point2Box.py --datapath [data folder] --loadmodel [LOADMODEL]
        ```
    - get Point net estimate 3D box using Lidar points: 
        ```
        python Point2Box.py --datapath [data folder] --loadmodel [LOADMODEL] --uselidar True
        ```
    
- Draw 3D box on left image: 
    ```
    python corner2box.py --datapath [DATAPATH] --uselidar [T/F]
    ```
- KITTI paper: http://www.cvlibs.net/publications/Geiger2013IJRR.pdf


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

Values |   Name   |   Description|
-------|----------|----------------------------------------------------------
   1  |  type      |   Describes the type of object: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram','Misc' or 'DontCare'
   1   | truncated  |  Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
   1  |  occluded   |  Integer (0,1,2,3) indicating occlusion state:  0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
   1   | alpha     |   Observation angle of object, ranging [-pi..pi]
   4   | bbox       |  2D bounding box of object in the image (0-based index): contains left, top, right, bottom pixel coordinates
   3    |dimensions |  3D object dimensions: height, width, length (in meters)
   3   | location   |  3D object location x,y,z in camera coordinates (in meters)
   1   | rotation_y  | Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1   | score       | Only for results: Float, indicating confidence in detection, needed for p/r curves, higher is better.

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

calib data we use:
  - T = 0.54
