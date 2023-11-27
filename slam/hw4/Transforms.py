import numpy as np
import numpy.linalg as li
from scipy.spatial.transform import Rotation

def rotatePoint(rodrigues, point):
    theta2 = np.dot(rodrigues, rodrigues)
    if (theta2 > np.finfo(np.float64).eps):
        theta = np.sqrt(theta2)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        theta_inv = 1.0 / theta
        w = theta_inv * rodrigues
        w_cross_pt = np.cross(w, point)
        tmp = np.dot(w, point) * (1 - cos_theta)
        result = point*cos_theta + w_cross_pt*sin_theta + w*tmp
    else:
        w_cross_pt = np.cross(rodrigues, point)
        result = point + w_cross_pt
    return result

def project(camera, point):

    Z = 2
    rot = camera[0:3]
    trans = camera[3:6]
    focal = camera[6]
    k1 = camera[7]
    k2 = camera[8]
    
    point = rotatePoint(rot, point) + trans
    
    # Compute distortion center
    xp = -point[0]/point[2]
    yp = -point[1]/point[2]
    
    # Compute distorted pixel point
    r2 = xp**2 + yp**2
    distortion = 1.0 + r2 * (k1 * k2 * r2)
    projection = -focal*distortion*np.array([xp, yp])
    #print(projection)
    return projection

def transform(camera, point):
    rot = camera[0:3]
    trans = camera[3:6]
    trans_point = rotatePoint(rot, point) + trans
    return trans_point