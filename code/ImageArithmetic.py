'''
加、减、乘、除
'''
import cv2
import numpy as np
def nothing(x):
    pass
img1=cv2.imread('lena.jpg')
img2=cv2.imread('rotate.jpg')
img=np.zeros_like(img2)
# img=np.zeros(img2.shape,np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('Trace','image',0,100,nothing)#轨道栏名称，窗体名，参数默认值，最大值，回调函数
sub=cv2.subtract(img2,img1)
cv2.imshow('substract',sub)
while(True):
    cv2.imshow('image',img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    r=cv2.getTrackbarPos('Trace','image')#获取滚动条的值
    r=float(r)/100
    img=cv2.addWeighted(img1,r,img2,1-r,0)

