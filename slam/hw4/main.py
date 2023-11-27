import numpy as np
import matplotlib as plt
import random
from scipy.spatial.transform import Rotation
from utils import *

def task1(x, y, K, Z=2):

    """ Generate the noise-free measurements of these 25 3D points in all views."""

    # get c5 3d points first, depth along the principal axis in the reference view is 2m
    C5_img = np.vstack((x, y))
    C5_img = np.vstack((C5_img, np.ones((1,x.shape[0]))))
    # print(C5_img)

    point_world = (np.linalg.inv(K) * Z @ C5_img).T
    # print(point_world)

    # relations in the reference view
    R_world = np.identity(3)
    t1 = [-0.1, 0.1, 0]
    t2 = [0, 0.1, 0]
    t3 = [0.1, 0.1, 0]
    t4 = [-0.1, 0, 0]
    t5 = [0, 0, 0]
    t6 = [0.1, 0, 0]
    t7 = [-0.1, -0.1, 0]
    t8 = [-0.1, 0, 0]
    t9 = [0.1, -0.1, 0]
    t_worlds = [t1,t2,t3,t4,t5,t6,t7,t8,t9]
    
    point_uv = []
    for t in t_worlds:
        #world = R_world @ point_world + np.asarray(t).reshape(3,-1) ########## -?
        world = point_world +t
        # print(world)
        uv = K *(1/Z) @ world.T
        uv = uv[0:2].T
        # print(uv,"\n")
        # print(world,"\n")
        point_uv.append(uv)

    return point_world, point_uv, t_worlds

def task2(point_world, point_uv, t_worlds):

    # Add noise to the x and y coordinates of all 2D measurements in the image plane by sampling from 
    # a normal distribution with a standard deviation of 0.25 pixels.
    sigma = 0.25 # standard deviation
    point_uv_noise= []
    for uv in point_uv:
        ##############cnm？？？？？？？？？reshape hai wo
        #print(uv,"\n")
        for pixel in uv:
            pixel[0] += random.gauss(0, sigma)
            pixel[1] += random.gauss(0, sigma)
        point_uv_noise.append(uv)
        # print(uv)

    # Perturb the 3D points to generate an initial guess for the 3D structure by adding noise to all 3 coordinates
    sigma = 0.02
    # print(point_world)
    for point in point_world:
        point[0] += random.gauss(0, sigma)
        point[1] += random.gauss(0, sigma)
        point[2] += random.gauss(0, sigma)
    point_world_noise = point_world
    # print(point_world_noise)

    # Perturb the camera poses to generate an initial guess for the views. 
    sigma_t = 0.002
    sigma_r = 0.002
    t_noise = []
    r_noise = []
    # R_noise = []
    for t in t_worlds:
        t_n = np.asarray(t)
        t_n[0] += random.gauss(0, sigma_t)
        t_n[1] += random.gauss(0, sigma_t)
        t_n[2] += random.gauss(0, sigma_t)
        t_noise.append(t_n)
        # print(t_n)
    for i in range(len(t_worlds)):
        r = np.asarray([random.gauss(0, sigma_r) for i in range(3)])
        r_noise.append(r)
        # print(R)
    
    return point_uv_noise, point_world_noise, r_noise, t_noise

def task3(point_uv_noise, point_world_noise, R_noise, t_noise, K):

    ba = BA(K, point_world_noise, point_uv_noise, R_noise, t_noise )
    camera_index, point_index, cameras, points, observations = ba.read_bal()
    ba.gauss_newton_BA_algebra()

    cameras = np.array(cameras, dtype=np.float64)
    print(cameras[:,0:6])
    points = np.array(points, dtype=np.float64)
    observations = -np.array(observations, dtype=np.float64)
    camera_index = np.array(camera_index, dtype=np.int8)
    point_index = np.array(point_index, dtype=np.int8)

    sba = SBA(cameras[:,0:6],points,observations,camera_index, point_index)
    camera_params, points_3d = sba.bundleAdjust()
    print(camera_params,"\n", points_3d)
    return camera_params, points_3d

if __name__ == '__main__':

    K = np.asarray([[500, 0, 0],
                    [0, 500, 0],
                    [0, 0, 1]])

    x, y = [],[]
    for i in [50, 185, 320, 455, 590]:
        for j in [40, 140, 240, 340, 440]:
            x.append(i)
            y.append(j)
    x = np.asarray(x)
    y = np.asarray(y)

    point_world, point_uv, t_worlds = task1(x,y,K)
    point_uv_noise, point_world_noise, r_noise, t_noise = task2(point_world, point_uv, t_worlds)
    # print(point_world_noise)
    camera_params, points_3d = task3(point_uv_noise, point_world_noise, r_noise, t_noise, K)
    visualize_world(points_3d,point_world_noise)



