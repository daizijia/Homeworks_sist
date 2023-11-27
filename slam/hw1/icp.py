import numpy as np
import open3d as o3d
import glob
import os
from preprocess import *
from utils import *
from kdTree import *
import time 
import random

def p2lICP(src, dst, init_trans = None, dst_normal = None, max_iterations = 20, tolerance = 0.001):
    """
        input: src, dst is N x 3 numpy array
        output: 4 x 4 transform matrix
    """
    if init_trans is not None:
        src = transpose(src, init_trans)

    num = min(src.shape[0], dst.shape[0])
    src = src[:num,:3]
    dst = dst[:num,:3]
    transform = np.eye(4)
    prev_error = 0
    src_prev = np.copy(src)
    if dst_normal is None:
        dst_normal = getNormals(dst, 30)###4 10 20 30 

    for i in range(max_iterations):
        knn_kdTree = KNNKdTree(n_neighbors=1)
        knn_kdTree.fit(dst)
        _,index = knn_kdTree.kneighbors(src)

        dst_in = dst[index[0]]
        normal = dst_normal[index[0]]

        transform,_,_ = best_p2l_transform(src, dst_in, normal) 

        error = p2lError(src, dst_in, normal) 
        print("error:" + str(error))

        src = transpose(src, transform)
        if np.abs(prev_error - error) < tolerance:
            break
        prev_error = error
    
    transform,_,_ = best_p2p_transform(src_prev, src)
    
    return transform

def p2pICP(s, d, init_trans = None, max_iterations = 50, tolerance = 0.001):

    num = min(s.shape[0], d.shape[0])
    s = s[:num,:3]
    d = d[:num,:3]

    m = s.shape[1]
    src = np.ones((m+1,s.shape[0]))
    dst = np.ones((m+1,d.shape[0]))
    src[:m,:] = np.copy(s.T)
    dst[:m,:] = np.copy(d.T)

    if init_trans is not None:
        src = np.dot(init_trans, src)

    prev_error = 0
    for i in range(max_iterations):
        knn_kdTree = KNNKdTree(n_neighbors=1)
        knn_kdTree.fit(dst[:m,:].T)
        dist,index = knn_kdTree.kneighbors(src[:m,:].T)
        transform,_,_ = best_p2p_transform(src[:m,:].T, dst[:m,index[0]].T)
        src = np.dot(transform, src)
        mean_error = np.mean(dist)
        print("error:" + str(mean_error))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    transform,_,_ = best_p2p_transform(s, src[:m,:].T)

    return transform

def buildMap(files, mode='p2l'):

    transforms = []
    # frame to frame
    former_xyz = None
    for file in files:
        print(file)
        pcd = o3d.io.read_point_cloud(file)
        pcd_xyz = np.asarray(pcd.points)

        if former_xyz is None:
            transform = np.eye(4)
            transforms.append(transform)
        else:
            if mode == 'p2l':
                transform = p2lICP(pcd_xyz, former_xyz)
            else:
                transform = p2pICP(pcd_xyz, former_xyz)
            transform = np.dot(transforms[-1],transform)###
            transforms.append(transform)
        former_xyz = pcd_xyz

    return transforms


def visualization(src,src_t,dst):
    # input:array
    src.paint_uniform_color([0.5,1,1])
    dst.paint_uniform_color([1,0.5,0.5])
    src_t.paint_uniform_color([1,1,0.5])
    o3d.visualization.draw_geometries([src,dst,src_t], width = 800, height =600)
    
def visualizationMap(files,c=None, transforms = None):
    pcds = o3d.geometry.PointCloud()
    if c is None:
        c = [[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)] for i in range(len(files))]

    if transforms is None:
        for i in range(len(files)):
            pcd = o3d.io.read_point_cloud(files[i])
            pcd.paint_uniform_color(c[i])
            pcds = pcds +pcd
    else:
        for i in range(len(files)):
            pcd = o3d.io.read_point_cloud(files[i])
            pcd.transform(transforms[i])
            pcd.paint_uniform_color(c[i])
            pcds = pcds +pcd
    o3d.io.write_point_cloud("p2presult.pcd",pcds)
    #o3d.visualization.draw_geometries([pcds], width = 800, height =600)

if __name__ == '__main__': 
    filepath = "/home/mpl/Desktop/slam_hw/hw1/datas/voxel_0.3"
    files = sorted(glob.glob(os.path.join(filepath, "*.xyz")))
    filepath_raw = "/home/mpl/Desktop/slam_hw/hw1/datas/raw"
    files_raw = sorted(glob.glob(os.path.join(filepath_raw, "*.xyz")))
    # files = files[10:15]
    # files_raw = files_raw[10:15]
    
    # unit test
    # src = o3d.io.read_point_cloud(files[4])
    # dst = o3d.io.read_point_cloud(files[5])
    # transform = p2lICP(np.asarray(src.points), np.asarray(dst.points))
    # # yellow is before,blue is after
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(src.points)
    # src.transform(transform)
    # visualization(src,point_cloud,dst)
    # print(transform)
    
    # build map
    c = [[random.uniform(0,1),random.uniform(0,1),random.uniform(0,1)] for i in range(len(files))]
    np.sort(c)
    visualizationMap(files,c)
    transforms = buildMap(files, mode = 'p2p')
    print(transforms)
    # visualizationMap(files,c, transforms)
    visualizationMap(files_raw,c, transforms)


    # # preprocess
    # i = 0
    # for file in files:
    #     print(file)
    #     pcd = o3d.io.read_point_cloud(file)
    #     pcd_xyz = np.asarray(pcd.points)
    #     #pcd_xyz_down = voxel_down_sample(pcd_xyz, voxel_size = 0.3)
    #     pcd_xyz_down = random_down_sample(pcd_xyz, radio =20)
    #     point_cloud = o3d.geometry.PointCloud()
    #     point_cloud.points = o3d.utility.Vector3dVector(pcd_xyz_down)
    #     path = "/home/mpl/Desktop/slam_hw/hw1/datas/random_20/" + str(i) + ".xyz"
    #     o3d.io.write_point_cloud(path, point_cloud)
    #     i += 1
    pass