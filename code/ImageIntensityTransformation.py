'''
线性变换、对数变换、伽马变换、彩色图像的线性变换、图像的正规化
'''
import cv2
import numpy as np
src=cv2.imread('lena.jpg',0)
cv2.imwrite('src.jpg',src)
negative=255-src
cv2.imwrite('negative.jpg',negative)
'''
线性变换是截断，不是取模
'''
# linear=np.subtract(np.multiply(src,2),100)
linear=src*float(2)-100

linear[linear>255]=255#数据截断
linear=np.around(linear).astype(np.uint8)
cv2.imshow('linear',linear)
cv2.imwrite('linear.jpg',linear)
#对数变换
log=np.log(1+src)
log=np.around(255*(log-np.min(log))/(np.max(log)-np.min(log)))
log=log.astype(np.uint8)#注意类型变换
cv2.imwrite("log transformation.jpg",log)
#伽马变换

power=np.power(src,2.5)
# power=np.around(255*(power-np.min(power))/(np.max(power)-np.min(power)))
# power=power.astype(np.uint8)#注意类型变换
#利用normalize进行归一化，等价于上面两条语句
power=cv2.normalize(power,power,255,0,cv2.NORM_MINMAX,cv2.CV_8U)
cv2.imshow('power transformation',power)
cv2.imwrite("power transformation.jpg",power)

power1=np.power(src,0.4)
power1=np.around(255*(power1-np.min(power1))/(np.max(power1)-np.min(power1)))
power1=power1.astype(np.uint8)#注意类型变换
cv2.imwrite("power1 transformation.jpg",power1)

piecewise=np.piecewise(src, [src<=100,(101<src)&(src<150),src>=150], [lambda src:float(0.5)*src,lambda src:src*float(1.5),lambda src:float(2.0)*src-50])
piecewise[piecewise>255]=255#数据截断
piecewise=np.around(piecewise).astype(np.uint8)
cv2.imwrite('piecewise.jpg',piecewise)

'''
彩色图像
'''
src=cv2.imread('lena.jpg',1)
cv2.imshow('color src',src)
dst=255-src
cv2.imshow('color dst',dst)
linear1=src*float(1.5)-100
linear1[linear1>255]=255#数据截断
linear1=np.around(linear1).astype(np.uint8)
cv2.imshow('color linear',linear1)

b,g,r=cv2.split(src)
b,g,r=255-b,255-g,255-r
dst=cv2.merge((b,g,r))
cv2.imshow('dst',dst)
#正规化函数
dst1=np.ones_like(src)
dst1=cv2.normalize(src,dst1,255,0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.imshow('normalize',dst1)


cv2.waitKey(0)
cv2.destroyAllWindows()