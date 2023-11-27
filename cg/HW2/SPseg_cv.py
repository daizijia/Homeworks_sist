"""use opencv"""
import cv2

img = cv2.imread("Datas/xiaokun.png")

slic = cv2.ximgproc.createSuperpixelSLIC(img,region_size=20,ruler = 20.0) #all 100 50 20 10 5 1
slic.iterate(10)     
 
mask = slic.getLabelContourMask() 
mask_inv = cv2.bitwise_not(mask)  
img_slic = cv2.bitwise_and(img,img,mask =  mask_inv) 

cv2.imwrite("Results/xiaokun_re.png",img_slic)


