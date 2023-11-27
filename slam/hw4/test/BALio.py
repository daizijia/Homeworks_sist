# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 19:07:35 2021

@author: Zijian Zhong

This script is to read data from BAL dataset
"""

from re import split as resplit

def read_bal(filename):
    
    """
    Read BAL dataset from txt file
    input:
        filename: txt filename
    output: 
        camera_index, point_index, cameras, points, observations
    """
    
    camera_index = [] # The camera index in each observation
    point_index = [] # The point index in each observation
    cameras = [] # Parameters of each camera (R(3x1 Rodrigues), t(3x1 Vector), f, k1, k2)
    points = []  # 3D Coordinates of each point (x, y, z)
    observations = [] # 2D Coordinates of each pixel (x, y) 
    with open(filename, 'r') as f:
        dataset_params = f.readline().split(' ')
        num_cams = int(dataset_params[0])
        print(num_cams)
        num_points = int(dataset_params[1])
        print(num_points)
        num_observations = int(dataset_params[2])
        print(num_observations)
        
        for i in range(num_observations):
            line = f.readline()
            line_split = resplit('[ ]+', line)
            camera_index.append(int(line_split[0]))
            point_index.append(int(line_split[1]))
            observations.append([float(line_split[2]), float(line_split[3])])
        
        for i in range(num_cams):
            cam_param = []
            for j in range(9):
                line = f.readline()
                cam_param.append(float(line))
            cameras.append(cam_param)
        
        for i in range(num_points):
            pt_param = []
            for j in range(3):
                line = f.readline()
                pt_param.append(float(line))
            points.append(pt_param)  
            
    return camera_index, point_index, cameras, points, observations
            