import numpy as np
import cv2

class Helper(object):
    """
    The fundamental matrix describes the relationship between two images which have similar feature points.
    It can be calculated using at least 8 points, therefore the 8 point algorithm
    """
    @staticmethod
    def calc_FundamentalMatrixFromPoints(R, t, K):
        t1, t2, t3 = [i for i in t]
        #print(t,t1,t2,t3)
        t_s = np.asarray([[0, -t3, t2],
                         [t3, 0, -t1],
                         [-t2, t1, 0]])
        E = t_s @ R
        F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

        return F
    
    @staticmethod
    def calc_FundamentalMatrixFromPoints_v2(img1,img2):
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        pts1 = []
        pts2 = []
        for i, (m,n) in enumerate(matches):
            if m.distance < 0.76*n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

        return F
    
    """
    Calculate epipolar line for a point on the other image, using the already calculated fundamental matrix.
    We calculate this by multiplying the Point (x,y) with the Fundamental Matrix. 
    Then solve for a line using the obtained solution.
    """
    @staticmethod
    def calc_EpipolarLineCoords(x, y, F, imageW):
        F = F.T
        point = np.matrix(np.reshape(np.array([x,y,1]),(3,1))) 
        solution = F * point
        a, b, c = float(solution[0]), float(solution[1]), float(solution[2])
        x1, x2 = 0, imageW
        if (b!=0):
            y1 = int(-((c/b) + ((a*x1)/b)))
            y2 = int(-((c/b) + ((a*x2)/b)))
        else:
            y1 = y2 = 0
        
        x1 += imageW
        x2 += imageW
        return x1, y1, x2, y2

    """
    Simple correlation between two patches of an image indicate how similar they are. This is not ZNCC.
    """
    @staticmethod
    def correlation(patch1, patch2):
        product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
        stds = patch1.std() * patch2.std()
        if stds == 0:
            return 0
        else:
            product /= stds
            return product
        
    @staticmethod
    def ssd(patch1, patch2):
        
        ssd = 0
        for i in range(patch1.shape[0]):
            for j in range(patch2.shape[1]):
                ssd += np.square(patch1[i][j]-patch2[i][j])
        #print("Ssd",ssd)
        return np.sum(ssd)
        pass