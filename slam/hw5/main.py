import numpy as np
import cv2
import glob
from scipy.spatial.transform import Rotation 
from skimage.draw import line
from Matching2d import Helper
from tqdm import *
import open3d as o3d

scale = 0.5
CameraIntrinsic = np.array([[517.3, 0, 318.6],
                            [0, 516.5, 255.3],
                            [0, 0, 1]])
CameraIntrinsic = CameraIntrinsic * scale
distort = [0.2624,-0.9531,-0.0054,0.0026,1.1633]

class Frame(object):
    def __init__(self):
        self.rotation = None
        self.translation = None
        self.img = None
        self.depth_map = None
        self.semi_dense_region = None
        pass

    def load_pose(self, pose):
        
        t = pose[1:4]
        qr = pose[4:8]
        r = Rotation.from_quat(qr) #########check
        R = r.as_matrix()
        self.rotation = R
        self.translation = t
        pass

    def load_image(self, filename):
        
        img = cv2.imread(filename)
        # add the rid distortion
        distCoeffs = np.float32(distort)
        img_undistored = cv2.undistort(img, CameraIntrinsic/scale, distCoeffs)
        img_undistored = img_undistored[10:-10,10:-10]
        cv2.imwrite("img_undistored.png",img_undistored)
        resized = cv2.resize(img_undistored, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        self.img = resized
        pass

    def detect_semi_dense_img(self):

        img = self.img
        img_denoise = cv2.GaussianBlur(img,(3,3),0) 
        kernel = np.ones((3,3),np.uint8)              
        #erosion = cv2.erode(img_denoise,kernel,iterations=1)  
        dilate = cv2.dilate(img_denoise,kernel,iterations=2) 
        img_edges = cv2.Canny(dilate, threshold1 = 60, threshold2 =200)
        #img_edges = cv2.Canny(img_denoise, threshold1 = 30, threshold2 =100)
        # print("img_edges",img_edges.shape)
        self.semi_dense_region = img_edges
        cv2.imwrite("semi.png",self.semi_dense_region)
        return img_edges

    def update_depth(self, filename):

        #depth = cv2.imread(filename,-1)
        #print("depth: ",depth)
        # self.depth_map = depth
        #cv2.imwrite("depth.png",self.depth_map)

        pass


def matching2d(img_a, img_b, R, t, img_a_semi):

    #img_a is ref

    match = Helper()
    #FMatrix = match.calc_FundamentalMatrixFromPoints(R, t, CameraIntrinsic)
    FMatrix = match.calc_FundamentalMatrixFromPoints_v2(img_a, img_b)
    FMatrix = FMatrix.T
    # FMatrix = np.asarray([[-2.33135115e-07, -6.12069632e-05,  6.09766706e-03],
    #                         [ 8.73860667e-05, -4.63159101e-06,  1.40340549e-02],
    #                         [-7.86699699e-03, -2.21686910e-02,  1.00000000e+00]]).T
    print("FMatrix:",FMatrix)
    imageWidth = img_a.shape[1]
    imageHeight = img_a.shape[0]

    point2d_a = []
    point2d_b = []
    pixels = []
    corr = []
    window_size = 5

    for i in range(imageHeight):
        for j in range(imageWidth):
            win = img_a[i-window_size:i+window_size, j-window_size:j+window_size]
            if img_a_semi[i][j] != 0 and win.shape[0] == window_size*2 and win.shape[1] == window_size*2:
                pixels.append([j,i])
                # print("win:",win.shape)

    #print("Pixels:",len(pixels))
    #lines = cv2.computeCorrespondEpilines(np.asarray(pixels).reshape(-1,1,2),1,FMatrix)
    #print("lines:",lines)
    
    for pix in tqdm(range(len(pixels))):
        #pix = pix+1
        x = pixels[pix][0]
        y = pixels[pix][1]
        x1, y1, x2, y2 = match.calc_EpipolarLineCoords(x, y, FMatrix, imageWidth)
        # print(" x1, y1, x2, y2:", x1, y1, x2, y2)
        color = tuple(np.random.randint(0,255,3).tolist())
        # cv2.line(img_b,(x1,y1),(x2,y2),color,300)

        # y1 = y1+20
        # y2 = y2+20
        pointsOnLine = []
        for i in range(1):
            y1_ = (y1-0+i)
            y2_ = (y2-0+i)
            pointsOnLine += list(zip(*line(*(x1,y1_), *(x2,y2_))))

        #print("pointsOnLine:",len(pointsOnLine))
        print("pointsOnLine", pointsOnLine[0],pointsOnLine[-1])
        filterWindowQuery = img_a[y-window_size:y+window_size, x-window_size:x+window_size]
        allFilterPointCoords = []
        for x_,y_ in pointsOnLine:
            if(imageHeight > y_ > 0):                 
                x_ -= imageWidth #locally in the image so start from 0
                filterWindowLine =img_b[y_-window_size:y_+window_size, x_-window_size:x_+window_size]
                if(filterWindowLine.shape[0] == window_size*2 and filterWindowLine.shape[1] == window_size*2):
                    allFilterPointCoords.append(((x_, y_), Helper.correlation(filterWindowQuery, filterWindowLine)))
                    #allFilterPointCoords.append(((x_, y_), Helper.ssd(filterWindowQuery, filterWindowLine)))

        allFilterPointCoords = sorted(allFilterPointCoords, key = lambda x: -x[1])
        #allFilterPointCoords = sorted(allFilterPointCoords, key = lambda x:x[1])
        
        if allFilterPointCoords and allFilterPointCoords[0][1] > 0.8:
            max_x, max_y = allFilterPointCoords[0][0]
            print(allFilterPointCoords[0][1])
            point2d_a.append([x, y])
            point2d_b.append([max_x, max_y])#########
            corr.append(allFilterPointCoords[0][1])
    print("point2d_b", len(point2d_b))

    for i in range(len(point2d_a)):
        # cv2.line(img_a,(x1,y1),(x2,y2),(20,120,50),5)
        cv2.circle(img_a, (point2d_a[i][0],point2d_a[i][1]), 1, (0,0,255), -1)
        cv2.circle(img_b, (point2d_b[i][0],point2d_a[i][1]), 1, (0,0,255), -1)
    cv2.imwrite("point2d_a.png",img_a)
    cv2.imwrite("point2d_b.png",img_b)
    return np.asarray(point2d_a), np.asarray(point2d_b),corr


def triangulation(pointl_vec,pointr_vec,R,t):
    n = pointl_vec.shape[0]
    pointl_cam_vec = []
    pointr_cam_vec = []
    for pointl in pointl_vec:
        pointl_cam_vec.append([(pointl[0] - CameraIntrinsic[0, 2]) / CameraIntrinsic[0, 0],(pointl[1] - CameraIntrinsic[1, 2]) / CameraIntrinsic[1, 1]])
    for pointr in pointr_vec:
        pointr_cam_vec.append([(pointr[0] - CameraIntrinsic[0, 2]) /CameraIntrinsic[0, 0],(pointr[1] - CameraIntrinsic[1, 2]) / CameraIntrinsic[1, 1]])
    T1 = np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]])
    T2 = np.concatenate((R,t.reshape(3,-1)),axis=1)
    # print(T1,T2)
    pointl_cam_vec = np.array(pointl_cam_vec).transpose()
    pointr_cam_vec = np.array(pointr_cam_vec).transpose()

    pts_4d = np.zeros((4,n))
    cv2.triangulatePoints(T1,T2,pointl_cam_vec,pointr_cam_vec,pts_4d)
    pts_3d = []
    for i in range(n):
        x = pts_4d[0,i]/pts_4d[3,i]
        y = pts_4d[1,i]/pts_4d[3,i]
        z = pts_4d[2,i]/pts_4d[3,i]
        pts_3d.append([x,y,z])
    pts_3d = np.array(pts_3d)

    return pts_3d

def get_relative_pose(r1,t1,r2,t2):
    # 1 is ref 
    T1 = np.eye(4)
    T2 = np.eye(4)
    T1[0:3,0:3] = r1
    T1[0:3,3:] = t1.reshape(-1,1)
    T2[0:3,0:3] = r2
    T2[0:3,3:] = t2.reshape(-1,1)
    T_ret_frame = np.linalg.inv(T1) @ T2
    T_ret_frame = T_ret_frame
    Rp = T_ret_frame[0:3,0:3]
    tp = T_ret_frame[0:3,3:].reshape(1,-1)[0]

    print("T:",T_ret_frame)
    print("Rp:",Rp)
    print("tp:",tp)
    return Rp, tp

def drawepilines(img1, img2, lines, pts1, pts2=None):
    print(img1.shape)
    r,c,_ = img1.shape 
    # img1_temp = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2_temp = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1_temp = cv2.line(img1,(x0,y0),(x1,y1),color,1)
        img1_temp = cv2.circle(img1_temp, tuple(pt1), 5, color, -1)
        img2_temp = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1_temp, img2_temp

def drawreconstruction_corr(img_a, img_b,point2d_a, point2d_b,corr):
    for i in range(len(point2d_a)):
        # cv2.line(img_a,(x1,y1),(x2,y2),(20,120,50),5)
        cv2.circle(img_a, (point2d_a[i][0],point2d_a[i][1]), 1, (0,0,155*corr[i]), -1)
        cv2.circle(img_b, (point2d_b[i][0],point2d_a[i][1]), 1, (0,0,155*corr[i]), -1)
    cv2.imwrite("point2d_acorr.png",img_a)
    cv2.imwrite("point2d_bcorr.png",img_b)

def drawreconstruction(img_a, point2d_a, depth):
    depth = np.abs(depth/np.max(depth))
    
    for i in range(len(point2d_a)):
        # cv2.line(img_a,(x1,y1),(x2,y2),(20,120,50),5)
        print(depth[i][0])
        cv2.circle(img_a, (point2d_a[i][0],point2d_a[i][1]), 1, (1*depth[i][0],55*depth[i][0],155*depth[i][0]), -1)

    cv2.imwrite("point2d_ar.png",img_a)


if __name__ == '__main__':
    img_filename = sorted(glob.glob("./data2/fig/*.png"))
    depth_filename = sorted(glob.glob("./data2/depth/*.png"))
    poses = np.loadtxt("./data/laoyanhunhua.txt",skiprows=1)
    
    frames = []
    for i in range(2):
        frame = Frame()
        frame.load_image(img_filename[i])
        frame.load_pose(poses[i])
        frame.detect_semi_dense_img()
        frames.append(frame)
        
    ref = frames.pop(1)
    for i in range(1):

        #check here
        Rp, tp = get_relative_pose(ref.rotation,ref.translation,frames[i].rotation,frames[i].translation)
        point2d_a, point2d_b,corr = matching2d(ref.img, frames[i].img, Rp, tp,ref.semi_dense_region)
        print(Rp,tp)
        print("mean corr",np.average(corr))

        point3d = triangulation(point2d_a, point2d_b, Rp, tp) #######
        if i == 0:
            points = point3d
        else:
            points = np.vstack((points,point3d))
        
        drawreconstruction_corr(ref.img, frames[i].img, point2d_a, point2d_b,corr)
        drawreconstruction(ref.img, point2d_a, point3d[:,2:3])


    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud("test.pcd",point_cloud)