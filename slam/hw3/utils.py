import cv2 as cv
import numpy as np
import glob
import os
from scipy import optimize as opt
import math



class Calibrator(object):

    def __init__(self, img_dir, shape_inner_corner = (7,8), size_grid= 0.033, visualization=True, scale = 0.5):

        self.img_dir = img_dir
        self.shape_inner_corner = shape_inner_corner
        self.size_grid = size_grid
        self.visualization = visualization
        self.scale = scale

        self.points_world = []
        self.points_pixel = []
        self.homography_matrices = []
        self.V = []
        self.b = None
        self.K = None 
        self.R = []
        self.t = []
        self.lamda = None

        # create the conner in world space
        w, h = shape_inner_corner
        cp_int = np.zeros((w * h, 3), np.float32)
        cp_int[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
        self.cp_world = cp_int * size_grid

        # images
        self.img_paths = []
        for extension in ["jpg", "png", "jpeg"]:
            self.img_paths += glob.glob(os.path.join(img_dir, "*.{}".format(extension)))
       
        assert len(self.img_paths), "No images for calibration found!"


    def getHomography(self, corners, n = 20):
        
        src = np.asarray(self.cp_world[: n])  # world
        dst = np.asarray(corners[: n])  # image

        P = np.zeros((2 * n, 9))

        i = 0
        for (srcpt, dstpt) in zip(src, dst):
            x, y, x_dash, y_dash = srcpt[0], srcpt[1], dstpt[0], dstpt[1]

            P[i][0], P[i][1], P[i][2] = -x, -y, -1
            P[i+1][0], P[i+1][1], P[i+1][2] = 0, 0, 0

            P[i][3], P[i][4], P[i][5] = 0, 0, 0
            P[i+1][3], P[i+1][4], P[i+1][5] = -x, -y, -1

            P[i][6], P[i][7], P[i][8] = x*x_dash, y*x_dash, x_dash
            P[i+1][6], P[i+1][7], P[i+1][8] = x*y_dash, y*y_dash, y_dash

            i = i+2

        _, _, vh = np.linalg.svd(P, full_matrices=True)
        h = vh[-1:]
        h.resize((3, 3))

        homography = h/h[2, 2]

        return homography


    def updateVMatrix(self,h):
        
        v_12 = [h[0][0]*h[0][1], (h[0][0]*h[1][1] + h[1][0]*h[0][1]), h[1][0]*h[1][1],
        (h[2][0]*h[0][1] + h[0][0]*h[2][1]), (h[2][0]*h[1][1] + h[1][0]*h[2][1]), h[2][0]*h[2][1]]

        
        v_1122 = [h[0][0]*h[0][0] - h[0][1]*h[0][1],
                2*(h[0][0]*h[1][0] - h[0][1]*h[1][1]),
                h[1][0]*h[1][0] - h[1][1]*h[1][1],
                2*(h[2][0]*h[0][0] - h[0][1]*h[2][1]),
                2*(h[2][0]*h[1][0] - h[1][1]*h[2][1]),
                h[2][0]*h[2][0] - h[2][1]*h[2][1]]

        self.V.append(v_12)
        self.V.append(v_1122)


    def getMatrixB(self):
        
        V = np.asarray(self.V)
        _, _, vh = np.linalg.svd(V, full_matrices=True)
        # solve Vb = 0 for b
        b = vh[-1:]
        return b


    def getCalibMatrix(self):
        
        b = self.b
        v = (b[0][1]*b[0][3] - b[0][0]*b[0][4])/(b[0][0]*b[0][2] - b[0][1]**2)
        lamda = b[0][5] - (b[0][3]**2 + v*(b[0][1]*b[0][3] - b[0][0]*b[0][4]))/b[0][0]
        alpha = math.sqrt(lamda/b[0][0])
        beta = math.sqrt(lamda*b[0][0]/(b[0][0]*b[0][2] - b[0][1]**2))
        gamma = (-1*b[0][1]*alpha**2*beta)/(lamda)
        u = (gamma*v)/beta - (b[0][3]*alpha**2)/lamda

        print("u = {}\nv = {}\nlamda = {}\nalpha = {}\nbeta = {}\ngamma = {}\n".format(
            u, v, lamda, alpha, beta, gamma))

        A = np.array([[alpha, gamma, u], [0, beta, v], [0, 0, 1]])
        return A, lamda
    
    def getExtrinsicParams(self,homography_matrix):
        
        lamda = self.lamda
        K = self.K

        K_inv = np.linalg.inv(K)

        r1 = np.dot(K_inv, homography_matrix[:, 0])
        lamda = np.linalg.norm(r1, ord=2),
        r1 = r1/lamda

        r2 = np.dot(K_inv, homography_matrix[:, 1])
        r2 = r2/lamda

        t = np.dot(K_inv, homography_matrix[:, 2])/(lamda)

        r3 = np.cross(r1, r2)

        R = np.asarray([r1, r2, r3])
        R = R.T

        return R, t

    def getReprojectionError(self,image_points, R, t):
        
        error = 0
        augment = np.zeros((3, 4))
        augment[:, :-1] = R
        augment[:, -1] = t

        N = np.dot(self.K, augment)

        for pt, wrldpt in zip(image_points, self.cp_world):
            M = np.array([[wrldpt[0]], [wrldpt[1]], [0], [1]])
            realpt = np.array([[pt[0]], [pt[1]], [1]])
            projpt = np.dot(N, M)
            projpt = projpt/projpt[2]
            diff = realpt - projpt
            error = error + np.linalg.norm(diff, ord=2)

        return error

    def calibrateCamera(self):
        
        w, h = self.shape_inner_corner

        for img_path in self.img_paths:
            img = cv.imread(img_path)
            #TODO: check
            img = cv.resize(img,None,fx=self.scale, fy=self.scale, interpolation = cv.INTER_CUBIC)
            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, corners = cv.findChessboardCorners(gray_img, (w, h), None)

            if ret:
                # print("True")
                # view the corners
                criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv.cornerSubPix(gray_img,corners,(11,11),(-1,-1),criteria)
                corners = corners2.reshape(-1, 2)

                if self.visualization:
                    cv.drawChessboardCorners(img, (w, h), corners, ret)
                    cv.imshow('FoundCorners', img)
                    cv.waitKey(500)

                corners = corners * (1/self.scale)
                homography_matrix = self.getHomography(corners)###########

                self.points_pixel.append(corners)
                self.homography_matrices.append(homography_matrix)
                self.updateVMatrix(homography_matrix)
        
        print("\nThe number of photos: \n{}".format(len(self.homography_matrices)))
        
        self.b = self.getMatrixB()

        self.K, self.lamda = self.getCalibMatrix()
        print("Initial estimate of Calibration matrix: \n\n{}".format(self.K))


        errorlist = []
        for image_points, homography_matrix in zip(self.points_pixel, self.homography_matrices):
            R, t = self.getExtrinsicParams(homography_matrix)
            reprojection_error = self.getReprojectionError(image_points, R, t)
            errorlist.append(reprojection_error/(w*h)) 
            self.R.append(R)
            self.t.append(t)

        print("Initial estimate of Extrinsic parameters: \nRotation Matrix: \n\n {} \n\nTransaltion Vector: \n\n {}".format(self.R, self.t))

        error = sum(np.asarray(errorlist))/(len(self.homography_matrices))##########
        print("\nMean Reprojection error: \n{}".format(error))
        
        return self.K, self.R, self.t, errorlist

    def calibrate(self):
        w, h = self.shape_inner_corner
        points_world = [] # the points in world space
        points_pixel = [] # the points in pixel space (relevant to points_world)
        for img_path in self.img_paths:
            img = cv.imread(img_path)
            img = cv.resize(img,None,fx=self.scale, fy=self.scale, interpolation = cv.INTER_CUBIC)

            gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            ret, cp_img = cv.findChessboardCorners(gray_img, (w, h), None)
            cp_img = cp_img * (1 / self.scale)
            if ret:
                # print("True")
                # cv2.cornerSubPix(gray_img, cp_img, (11,11), (-1,-1), criteria)
                points_world.append(self.cp_world)
                points_pixel.append(cp_img)
                
                # view the corners
                if self.visualization:
                    cv.drawChessboardCorners(img, (w, h), cp_img, ret)
                    cv.imshow('FoundCorners', img)
                    cv.waitKey(500)

        # calibrate the camera
        ret, mat_intri, coff_dis, v_rot, v_trans = cv.calibrateCamera(points_world, points_pixel, gray_img.shape[::-1], None, None)

        print ("ret: {}".format(ret))
        print ("intrinsic matrix: \n {}".format(mat_intri))
        # in the form of (k_1, k_2, p_1, p_2, k_3)
        print ("distortion cofficients: \n {}".format(coff_dis))
        print ("rotation vectors: \n {}".format(v_rot))
        print ("translation vectors: \n {}".format(v_trans))

        # calculate the error of reproject
        total_error = 0
        for i in range(len(points_world)):
            points_pixel_repro, _ = cv.projectPoints(points_world[i], v_rot[i], v_trans[i], mat_intri, coff_dis)
            error = cv.norm(points_pixel[i], points_pixel_repro, cv.NORM_L2) / len(points_pixel_repro)
            total_error += error
        print("Average error of reproject: {}".format(total_error / len(points_world)))

        self.mat_intri = mat_intri
        self.coff_dis = coff_dis

        return mat_intri, coff_dis




