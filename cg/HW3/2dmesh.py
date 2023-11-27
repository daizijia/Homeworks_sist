'''
@Author:Dai ZiJia
for cg homework
'''
import cv2
import numpy as np
from scipy.spatial import Delaunay
from pointsim import *
from ZoomIn import *

route = "data/"
filename = "cat.png"
img = cv2.imread(route + filename)
print(img.shape)

#generate random points
pointNumber = 10001
points = generate_uniform_random_points(img, n_points = pointNumber)

pts = np.zeros((points.shape[0], 2))
pts[:, 0] = points[:, 1]
pts[:, 1] = points[:, 0]
points = pts

# for initial
# points = np.zeros((pointNumber, 2))
# points[:, 0] = np.random.randint(1, img.shape[0], pointNumber)
# points[:, 1] = np.random.randint(1, img.shape[1], pointNumber)
#print(points)

tri = Delaunay(points)
#print(tri.simplices)
center = np.sum(points[tri.simplices], axis = 1) / 3.0
color = np.array([img[int(x)][int(y)] for x, y in center])

gray_img = img[:,:,0]
gray = np.array([gray_img[int(x)][int(y)] for x, y in center])
#print(gray)

t=''
with open ('pointsMesh.txt','w') as q:
    cnt = 0
    for i in points:
        t=t+'v'+' '
        for e in range(len(points[0])):
            t=t+str(i[e])+' '
        t=t+str(gray[cnt])+' '
        q.write(t.strip(' '))
        q.write('\n')
        t=''
        cnt += 1
    for i in tri.simplices:
        t=t+'f'+' '
        for e in range(len(tri.simplices[0])):
            t=t+str(i[e])+' '
        q.write(t.strip(' '))
        q.write('\n')
        t=''

tri_mesh = np.ones((img.shape[0], img.shape[1], 3), np.uint8) * 255

for i in range(color.shape[0]):
    index = tri.simplices[i]
    pt1 = (int(points[index[0]][1]), int(points[index[0]][0]))
    pt2 = (int(points[index[1]][1]), int(points[index[1]][0]))
    pt3 = (int(points[index[2]][1]), int(points[index[2]][0]))
    triangle_cnt = np.array( [pt1, pt2, pt3])
    cv2.drawContours(tri_mesh, [triangle_cnt], 0, tuple([int(x) for x in color[i]]), -1)
    #cv2.polylines(tri_mesh, [triangle_cnt], isClosed = True, color = (128, 138, 135), thickness = 1)

cv2.imwrite("result/" + filename[:-4] + str(pointNumber) + ".png", tri_mesh)

zoom_in(10,img,points)


