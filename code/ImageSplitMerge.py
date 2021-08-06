'''
分离、融合
'''
import cv2
import numpy as np
img=cv2.imread("lena.jpg")
b,g,r=cv2.split(img)
cv2.imshow("b",b)
cv2.imshow("g",g)
cv2.imshow("r",r)
print(img.shape)
src=img
b1,g1,r1=cv2.split(src)
zeros = np.zeros(src.shape[:2],dtype=np.uint8)#创建与源图片大小相同的数组，全部设置为0
#融合三通道回RGB图片，因为只想分别显示各通道的图片，所以除了要显示的通道外，其余两个通道均用0。
print(zeros.shape)
b=cv2.merge([b1,zeros,zeros])
g=cv2.merge([zeros,g1,zeros])
r=cv2.merge([zeros,zeros,r1])
cv2.imshow("b1",b)
cv2.imshow("g1",g)
cv2.imshow("r1",r)
dst=cv2.merge((b1,g1,r1))#参数为元组
cv2.imshow('merge',dst)
cv2.waitKey(0)
