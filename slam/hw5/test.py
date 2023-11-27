import numpy as np 
import cv2 
from matplotlib import pyplot as plt

# 读入图片
img1 = cv2.imread("data2/fig/1305031107.911541.png", 0)
img2 = cv2.imread("data2/fig/1305031108.343278.png", 0)

scale = 0.5
img1 = cv2.resize(img1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
# 定义SIFT算子
sift = cv2.SIFT_create()

# 找到特征点和计算特征
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN 参数
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

# 选择匹配点中匹配度比较好的点
pts1 = []
pts2 = []
for i, (m,n) in enumerate(matches):
    if m.distance < 0.76*n.distance:
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

# 类型转换
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# 找到基本矩阵
F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

# 选择内部的点
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

## 画极线
def drawepilines(img1, img2, lines, pts1, pts2):
    r,c = img1.shape 
    img1_temp = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_temp = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1_temp = cv2.line(img1_temp,(x0,y0),(x1,y1),color,1)
        img1_temp = cv2.circle(img1_temp, tuple(pt1), 5, color, -1)
        img2_temp = cv2.circle(img2_temp, tuple(pt2), 5, color, -1)
    return img1_temp, img2_temp

print(F)
# F = np.asarray([[ 6.60418518e-08, -5.04442990e-06,  1.68863398e-03],
#  [4.76564529e-06, -8.58951401e-08, -1.37843760e-03],
#  [-2.21676785e-03,  1.37575138e-03,  1.88862308e-01]])
# F = F.T

# 根据右图的点找到左图的极线
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2),2,F)
lines1 = lines1.reshape(-1,3)
img5, img6 = drawepilines(img1,img2,lines1,pts1,pts2)

# 根据左图的点找到右图的极线
lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2),1,F)
lines2 = lines2.reshape(-1,3)
img3, img4 = drawepilines(img2,img1,lines2,pts2,pts1)

res = np.hstack((img3, img5))
cv2.imwrite("res.png", res)


# res1 = np.hstack((img5,img6))
# res2 = np.hstack((img4,img3))

# cv2.imshow("res1", res1)
# cv2.imshow("res2", res2)


# cv2.waitKey(0)
# cv2.destroyAllWindows()
