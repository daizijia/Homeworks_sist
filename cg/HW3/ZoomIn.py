import cv2
import numpy as np
from scipy.spatial import Delaunay

def zoom_in(scale,img, points):
    pts = scale * points
    tri = Delaunay(pts)
    tri_mesh = np.ones((img.shape[0] * scale, img.shape[1] * scale, 3), np.uint8) * 255 
    center = np.sum(pts[tri.simplices], axis = 1) / (3.0 * scale )
    print(center)
    color = np.array([img[int(x)][int(y)] for x, y in center])
    for i in range(color.shape[0]):
        index = tri.simplices[i]
        pt1 = (int(pts[index[0]][1]), int(pts[index[0]][0]))
        pt2 = (int(pts[index[1]][1]), int(pts[index[1]][0]))
        pt3 = (int(pts[index[2]][1]), int(pts[index[2]][0]))
        triangle_cnt = np.array( [pt1, pt2, pt3])
        cv2.drawContours(tri_mesh, [triangle_cnt], 0, tuple([int(x) for x in color[i]]), -1)
        #cv2.polylines(tri_mesh, [triangle_cnt], isClosed = True, color = (128, 138, 135), thickness = 1)
    cv2.imwrite("result/zoomin" + str(scale) + ".png", tri_mesh)
