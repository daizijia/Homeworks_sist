import cv2
import numpy as np
import os
import glob
from utils import *
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation

dir = 'partB'

aK = np.asarray([[1.46134765e+03, 1.50142710e+00, 1.23668551e+03],
                [0.00000000e+00, 1.45997918e+03, 1.05589039e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

bK = np.asarray([[1.44267147e+03, 5.78753745e+00, 1.21903022e+03],
                [0.00000000e+00, 1.43890494e+03, 1.04461466e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

def get_image(dir,choose = 'A'):
    img_paths = []
    img_paths += glob.glob(os.path.join(dir, "{}*.png".format(choose)))
    img_paths.sort()
    return img_paths

def get_pose(pose_dir,choose = 'A'):
    pose_paths = []
    pose_paths += glob.glob(os.path.join(pose_dir, "hw3_v2_{}*.pose".format(choose)))
    pose_paths.sort()

    R = [] ########not euler
    t = []
    Homos = []
    T = np.eye(4)
    for p in pose_paths:
        # print(p)
        with open(p, "r",encoding="utf-8") as f:
            T = np.eye(4)
            data = f.readlines()
            pose =  data[1]
            pose = pose.split(" ")
            qx,qy,qz,qw = [float(i) for i in pose[0:4]]
            rr = np.asarray([[2*(qw**2 + qx**2)-1, 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qx)],
                             [2*(qx*qy + qw*qz), 2*(qw**2 + qy**2)-1, 2*(qy*qz - qw*qx)],
                             [2*(qx*qz + qw*qx), 2*(qy*qz + qw*qx), 2*(qw**2 + qz**2)-1]])
            # print(rr)
            # rr = Rotation([qx,qy,qz,qw]).as_matrix()
            tt = np.array(pose[4:7],dtype="float")
            T[:3,:3] = rr
            T[:3, 3] = tt
            T = np.linalg.inv(T)
            # print(T)
            Homos.append(T)
                # transforms = get_frame_trans_B(Homos)
    for RT in Homos: ############################
        R.append(RT[:3,:3])
        t.append(RT[:3, 3].reshape((3,1)))

    return np.asarray(R),np.asarray(t)

def get_image(dir,choose = 'A'):
    img_paths = []
    img_paths += glob.glob(os.path.join(dir, "{}*.png".format(choose)))
    img_paths.sort()
    return img_paths

end_to_base_Ra, end_to_base_ta = get_pose(dir, choose = 'A')
end_to_base_Rb, end_to_base_tb = get_pose(dir, choose = 'B')

imga = get_image(dir, choose = 'A')
imgb = get_image(dir, choose = 'B')

imga = get_image(dir, choose = 'A')
imgb = get_image(dir, choose = 'B')

pattern_size = (8, 7) 
square_size = 0.033 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


object_points = []
image_points = []


scale = 0.5
#while len(image_points) < 20:

for img_path in imgb:  
        img=cv2.imread(img_path)
        img = cv.resize(img,None,fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        corners = corners * (1 / scale)
        if ret:
            object_point = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            object_point[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            object_point *= square_size

            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            object_points.append(object_point)
            image_points.append(corners_refined)

camera_matrix = bK ########
dist_coeffs = None
object_points_array = np.array(object_points)

rvecs_camera = []
tvecs_camera = []
for i in range(len(image_points)):

    success, rvec, tvec = cv2.solvePnP(
        object_points_array[i],
        image_points[i],
        camera_matrix,
        dist_coeffs)
    rvecs_camera.append(rvec)
    tvecs_camera.append(tvec)
    # print(tvec)

R2, T2 = cv2.calibrateHandEye(
    end_to_base_Ra, 
    end_to_base_ta,
    np.array(rvecs_camera),
    np.array(tvecs_camera),
    method=cv2.CALIB_HAND_EYE_TSAI)
RT2=np.column_stack((R2,T2))
RT2 = np.row_stack((RT2, np.array([0, 0, 0, 1])))
print(f'Hand-eye transformation:\n{RT2}')

