import numpy as np
import open3d as o3d
from kdTree import *

def transpose(xyz, trans):

    R = trans[:3, :3]
    t = trans[:3, 3]
    xyz_trans = (R @ xyz.T).T + t
    return xyz_trans

def getNormals(xyz, k = 4):
    
    knn_kdTree = KNNKdTree(n_neighbors=4)
    knn_kdTree.fit(xyz)
    _,index = knn_kdTree.kneighbors(xyz)
    number = xyz.shape[0]

    normals = np.zeros((number,3))
    for i in range(number):
        nn = xyz[index[1:,i]]
        c = np.cov(nn.T)
        w,v = np.linalg.eig(c)
        normals[i] = v[:,np.argmin(w)]

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    # pcd.normals = o3d.utility.Vector3dVector(normals)
    # o3d.visualization.draw_geometries([pcd],point_show_normal=True,width = 800, height =600)
    return normals

def getNormals_o3d(xyz, k = 30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=k))
    # o3d.visualization.draw_geometries([pcd],point_show_normal=True,width = 800, height =600)
    return np.asarray(pcd.normals)


def p2lError(src, dst, normals = None):
    
    if normals is None:
        normals = getNormals(src, 4)
    
    error = 0 
    for i in range(src.shape[0]):
        e = np.dot((src[i]-dst[i]).T, normals[i])
        error = error + np.abs(e)

    return error/src.shape[0]

def cross_op(r):
    R = np.zeros((3, 3))
    R[0, 1] = -r[2]
    R[0, 2] = r[1]
    R[1, 2] = -r[0]
    R = R - R.T
    return R

def best_p2l_transform(src, dst, normals):
    # reject
    # angles = np.arccos(np.einsum('ij,ij->i', src, normals) / np.linalg.norm(src, axis=1)*np.linalg.norm(normals, axis=1))
    # index = np.bitwise_or(angles <= np.pi/4, angles >= 3*np.pi/4) # due to ambiguous orientaion
    # src = src[index]
    # dst = dst[index]
    # normals = normals[index]

    cross = np.cross(src, normals)
    Para = np.append(normals, cross, axis=1)  # n*6
    b = np.sum(((src - dst) * normals), axis=1) # n*1

    b = np.dot(np.array([b]), Para).T
    A = np.dot(Para.T, Para)
    delta_translation_rotation = np.linalg.solve(A, -b).T[0]

    t = delta_translation_rotation[:3]
    r = delta_translation_rotation[3:]
    theta = np.linalg.norm(r, 2)
    k = r / theta
   
    R = np.cos(theta)*np.eye(3)+np.sin(theta)*cross_op(k)+(1-np.cos(theta))*np.outer(k, k)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T, R, t
 
def best_p2p_transform(src, dst):

    m = src.shape[1]
    centroid_A = np.mean(src, axis=0)
    centroid_B = np.mean(dst, axis=0)
    AA = src - centroid_A
    BB = dst - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    t = centroid_B.T - np.dot(R,centroid_A.T)
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

def ransac_rid_plane(xyz):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    _,inliers = pcd.segment_plane(distance_threshold = 1, ransac_n = 1000, num_iterations = 100)
    pcd_out = pcd.select_by_index(inliers,invert = True)
    return np.asarray(pcd_out.points)

# def kneighbors(k, src, dst):
#     dist = np.zeros((src.shape[0], dst.shape[0])) 
#     M = np.dot(src, dst.T)
#     te = np.square(src).sum(axis=1)
#     tr = np.square(dst).sum(axis=1)
#     dist = np.sqrt(np.abs(-2 * M + tr + np.matrix(te).T)) 
#     sorted_distances = dist.argsort() 
#     k_neighbors = sorted_distances[0:k]
#     # print(k_neighbors)
#     return np.sort(dist)[0:k], k_neighbors
#     # assert src.shape == dst.shape
#     # neigh = NearestNeighbors(n_neighbors=1)
#     # neigh.fit(dst)
#     # distances, indices = neigh.kneighbors(src, return_distance=True)
#     # return distances.ravel(), indices.ravel()
