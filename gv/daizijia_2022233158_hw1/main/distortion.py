import numpy as np
import os
import glob
import cv2 as cv

def get_points(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    
    points = []
    for i in range(len(lines)-1):
        curline = np.array([float(x) for x in lines[i].split(',')])
        points.append(curline[:])
    
    return np.asarray(points)

def get_lamda(x0,y0,a,b,c):

    lamda = 1 / (x0 ** 2 + y0 ** 2 + a * x0 + b * y0 + c)
    print("lamda: ",lamda)
    return lamda

def get_x0y0(ABC_raw):
    """Only use 3 ABC
       for image 1 select line: 3,2,6
       for image 2 select line: 
    """

    ABC = []
    ABC.append(ABC_raw[0])
    ABC.append(ABC_raw[6])
    ABC.append(ABC_raw[14])
    M = np.zeros((2, 3))
    M[0:1,:] = ABC[0]-ABC[1]
    M[1:2,:] = ABC[0]-ABC[2]
    # M[2:3,:] = ABC[1]-ABC[2]
    A = M[:2,:2]
    b = - M[:2,2:]
    # TODO: double check
    ans = np.linalg.solve(A,b)
    print("ans",ans)
    y0, x0 = ans[0][0], ans[1][0]
    x0, y0 = ans[0][0], ans[1][0]
    return x0, y0

def get_ABC(points):
    """
    points: n x 2 array
    M: n x 3 array
    """

    M = np.hstack((points, np.ones((points.shape[0],1))))
    b = -(np.power(M[:,0], 2) + np.power(M[:,1], 2))

    A_B_C = np.linalg.inv(M.T @ M) @ M.T @ b
    # normalize_scale = np.sqrt(A_B_C[0] ** 2 + A_B_C[1] ** 2)
    A_B_C = A_B_C#.T #/ normalize_scale
    A_B_C = A_B_C.reshape(1,3)

    return A_B_C

def undistort(img, x0, y0, lamda):
    img_undistort = np.zeros((img.shape[0],img.shape[1],3), np.uint8)
    img_undistort = np.zeros((img.shape[0]+240,img.shape[1]+240,3), np.uint8)

    print("x0,y0,lamda: ",x0, y0, lamda)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            y_d = i
            x_d = j
            r_d = np.sqrt((x_d-x0) **2 + (y_d-y0) **2)
            temp = (1 + lamda * r_d **2)
            x_u = x0 + (x_d-x0) / temp 
            y_u = y0 + (y_d-y0) / temp
            x_u = x0 + (x_d-x0) / temp + 120
            y_u = y0 + (y_d-y0) / temp + 120
            img_undistort[int(y_u)][int(x_u)] = img[y_d][x_d]

    cv.imwrite('/home/daizj/Homeworks/gv/CS288_Geometric_Vision_HW01/result/undistort.png',img_undistort)

def plot_line(img, lines):

    font = cv.FONT_HERSHEY_SIMPLEX  
    font_scale = 1  
    color = (255, 255, 0)  
    thickness = 2  

    for i in range(len(lines)):
        for point in lines[i]:
            
            cv.putText(img, str(i+1), (int(point[0]),int(point[1])), font, font_scale, color, thickness)

    cv.imwrite('/home/daizj/Homeworks/gv/CS288_Geometric_Vision_HW01/data/image1_line.png',img)
    

if __name__ == '__main__':
    flag = 2
    if flag == 1:
        path = 'data/image1/'
        imagepath = '/home/daizj/Homeworks/gv/CS288_Geometric_Vision_HW01/data/image1.png'
    else:
        path = 'data/image2/'
        imagepath = '/home/daizj/Homeworks/gv/CS288_Geometric_Vision_HW01/data/image2.png'

    txtlist = sorted(glob.glob(path+'*.txt'))
    img = cv.imread(imagepath)
    h, w, _ = img.shape

    lines = []
    ABC = []
    for linepath in txtlist:
        line = get_points(linepath)
        lines.append(line)
        ABC.append(get_ABC(line))
    
    # plot_line(img, lines)
    if flag != 1:
        x0, y0 = get_x0y0(ABC)
        # x0, y0 =390,310
    else:
        x0, y0 = w/2, h/2
    
    lamda = get_lamda(x0, y0, ABC[0][0][0], ABC[0][0][1], ABC[0][0][2])
    # lamda = 1e-6
    undistort(img, x0, y0, lamda)
    
    
