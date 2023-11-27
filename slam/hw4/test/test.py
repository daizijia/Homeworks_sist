# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 19:40:23 2021

@author: Zijian Zhong
"""

import numpy as np
import numpy.linalg as li
import BALio
import Transforms
import Visualize
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

bal_filename = "test/problem-49-7776-pre.txt" # The BAL dataset file
processors = 4 # Define your processor cores
camera_index, point_index, cameras, points, observations = BALio.read_bal(bal_filename)

# proj0 = observations[0]
# cam0 = cameras[camera_index[0]]
# pt0 = points[point_index[0]]
cameras = np.array(cameras, dtype=np.float64)
points = np.array(points, dtype=np.float64)
observations = np.array(observations, dtype=np.float64)

num_cameras = len(cameras)
num_points = len(points)
num_observations = len(observations)




epochs = 20
for epoch in range(epochs):
    cost = 0
    # J = np.zeros((6*num_cameras+3*num_points, 2*num_observations), dtype=np.float64) # Cannot store dense jacobian like this -> 80GB memory needed
    J = csr_matrix((np.zeros(1), (np.zeros(1), np.zeros(1))),shape=(2*num_observations, 6*num_cameras+3*num_points))
    J = lil_matrix(J)
    b = np.zeros(6*num_cameras+3*num_points)
    for i in range(num_observations):
        observed = observations[i]
        print(observed)
        cam_idx = camera_index[i]
        pts_idx = point_index[i]
        camera = cameras[cam_idx]
        point = points[pts_idx]
        projected = Transforms.project(camera, point)
        print(projected)
        # Calc Jacobian
        # Camera Jacobian
        fx = camera[6]
        fy = fx
        p3d = Transforms.transform(camera, point)
        X = p3d[0]
        Y = p3d[1]
        Z = p3d[2]
        Z2 = Z**2
        JT = -np.array([
            [fx/Z, 0, -fx*X/Z2, -fx*X*Y/Z2, fx+fx*X**2/Z2, -fx*Y/Z],
            [0, fy/Z, -fy*Y/Z2, -fy-fy*Y**2/Z2, fy*X*Y/Z2, fy*X/Z]
            ])
        J[2*i : 2*i+2, cam_idx*6 : cam_idx*6+6] = JT
        
        # Point Jacobian
        JP = -np.array([
            [fx/Z, 0, -fx*X/Z2],
            [0, fy/Z, -fy*Y/Z2]
            ])
        J[2*i : 2*i+2, num_cameras*6 + pts_idx*3 : num_cameras*6 + pts_idx*3 + 3] = JP
        residual = observed - projected
        b[cam_idx*6 : cam_idx*6 + 6] += -np.dot(JT.T, residual)
        b[num_cameras*6 + pts_idx*3 : num_cameras*6 + pts_idx*3 + 3] += -np.dot(JP.T, residual)
        cost += li.norm(residual)
        # print(cost)
        if i%1000 == 0:
            print(f"iter: {i} in iters {num_observations}")
    # break
    H = J.T @ J
    update = spsolve(H, b)
    cam_update = np.reshape(update[0:6*num_cameras], (num_cameras, 6))
    pts_update = np.reshape(update[6*num_cameras:], (num_points, 3))
    cameras[:,0:6] += cam_update
    points += pts_update
    print(f"Epoch {epoch} Total cost:", cost)