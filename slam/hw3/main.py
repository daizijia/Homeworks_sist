import cv2
import numpy as np
import os
import glob
from utils import *
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import EPnP
img_dira = 'partA/chooseA'
img_dirb = 'partA/chooseB'
dir = 'partB'

aK = np.asarray([[1.46134765e+03, 1.50142710e+00, 1.23668551e+03],
                [0.00000000e+00, 1.45997918e+03, 1.05589039e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

bK = np.asarray([[1.44267147e+03, 5.78753745e+00, 1.21903022e+03],
                [0.00000000e+00, 1.43890494e+03, 1.04461466e+03],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
def visualize_calib(K,R,t):
    # visualize the calibration
    # matlab?
    w,h = [K[1][2],K[0][2]]
    f = K[0][0]
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.scatter([0],[0],[0],s = 200,c = "k")
    for i in range(len(R)):
        xyz = np.asarray([[0,0,0],
                       [0,0,h],
                       [w,0,h],
                       [w,0,0],
                       [0,0,0]])
        xyz_trans = (R[i] @ xyz.T).T + f * t[i].T + np.asarray([0,f,0]).T

        x,y,z = [i for i in xyz_trans.T]
        ax.plot(x,y,z)
    plt.show()

def visualize_errors(errors):
    # visualize the reprojection errors
    x = ["p{}".format(i+1) for i in range(len(errors))]
    y = [ i for i in errors]
    maximum = max(errors)
    for i in range(len(errors)):
        if errors[i] == maximum:
            plt.bar(x[i],y[i],color = 'indianred')
        else:
            plt.bar(x[i],y[i],color = 'c')

    plt.axhline(np.average(np.asarray(errors)),c = 'g',ls = '-.')
    plt.title("Reprojection Errors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    pass

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
            # T = np.linalg.inv(T) ##############################
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


def chess_to_cam(imgs,camera_matrix):
    pattern_size = (8, 7) 
    square_size = 0.033 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points = []
    image_points = []
    scale = 0.5

    flag = 0
    for img_path in imgs:  
        flag+=1
        img=cv2.imread(img_path)
        img = cv.resize(img,None,fx=scale, fy=scale, interpolation = cv.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        if ret:
            # cv.drawChessboardCorners(img, (8, 7), corners, ret)
            # cv.imshow('FoundCorners', img)
            # cv.waitKey(500)
            # cv.imwrite("{}.png".format(flag),img)
            corners = corners * (1 / scale)
            object_point = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            object_point[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
            object_point *= square_size

            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points.append(object_point)
            image_points.append(corners_refined)

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
        
        # epnp = EPnP.EPnP()
        # error, Rt, Cc, Xc = epnp.efficient_pnp(object_points_array[i], image_points[i], camera_matrix)
        # print(Rt)
        rvecs_camera.append(rvec)
        tvecs_camera.append(tvec)
        
    return np.asarray(rvecs_camera), np.asarray(tvecs_camera)


def main_partA():
    calib = Calibrator(img_dirb,visualization=False,scale=0.5)
    # calib.calibrate()
    K,R,t,errors = calib.calibrateCamera()
    visualize_calib(K,R,t)
    visualize_errors(errors)

def main_partB():

    #a hands , b cameras 
    end_to_base_Ra, end_to_base_ta = get_pose(dir, choose = 'A')
    end_to_base_Rb, end_to_base_tb = get_pose(dir, choose = 'B')

    imga = get_image(dir, choose = 'A')
    imgb = get_image(dir, choose = 'B')

    chess_to_cam_Ra,chess_to_cam_ta = chess_to_cam(imga,aK)
    chess_to_cam_Rb,chess_to_cam_tb = chess_to_cam(imgb,bK)

    
    R1,T1 = cv2.calibrateHandEye(end_to_base_Ra, end_to_base_ta, 
                                 chess_to_cam_Rb,chess_to_cam_tb)##########
    RT1=np.column_stack((R1,T1))
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    print("RT1:\n",RT1)

    R2,T2 = cv2.calibrateHandEye(end_to_base_Rb, end_to_base_tb,
                                 chess_to_cam_Ra,chess_to_cam_ta)##########
    RT2=np.column_stack((R2,T2))
    RT2 = np.row_stack((RT2, np.array([0, 0, 0, 1])))
    print("RT2:\n",RT2)

    print(RT1@np.linalg.inv(RT2))
    
    pass

if __name__ == '__main__':
    #main_partA()
    main_partB()

    pass
