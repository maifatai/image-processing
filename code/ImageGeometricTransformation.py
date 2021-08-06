'''
图像的尺寸变换、旋转、平移、镜像、仿射变换、投影变换、极坐标变换
'''
import cv2
import numpy as np
src=cv2.imread('lena.jpg')
'''
计算仿射变换矩阵，有6个未知数[[a11,a12,a13],[a21,a22,a23],[0,0,1]],可以通过三个点计算
当进行多个变换时，只需要利用np.dot()函数将相应的仿射变换矩阵相乘即可。
'''
set1=np.array([[0,0],[200,0],[0,200]],dtype=np.float32)
set2=np.array([[0,0],[100,0],[0,100]],np.float32)
mat_affine=cv2.getAffineTransform(set1,set2)
print(mat_affine)
'''
图像尺寸变换
'''
shape=tuple((np.array(src.shape[:2])/2).astype(int))
reduce=cv2.resize(src,shape)
cv2.imshow('reduce',reduce)

mat=np.array([[0.5,0,0],[0,0.5,0]],np.float32)
reduce1=cv2.warpAffine(src,mat,src.shape[:2])#第三行默认是[0,0,1]
cv2.imshow('reduce1',reduce1)

shape=tuple((np.array(src.shape[:2])*4).astype(int))
enlarge=cv2.resize(src,shape,interpolation=cv2.INTER_CUBIC)
cv2.imshow('enlarge',enlarge)

fix_size=cv2.resize(src,(300,500))
cv2.imshow('fix size',fix_size)

scale=cv2.resize(src,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_AREA)
cv2.imshow('scale',scale)

#
'''
图像旋转
'''
center=tuple((np.array(src.shape[:2])/2))
mat=cv2.getRotationMatrix2D(center,45,1)#旋转仿射变换矩阵，第一个参数旋转中心，第二个参数旋转角度，第三个参数缩放比例
rotate=cv2.warpAffine(src,mat,src.shape[:2],flags=cv2.INTER_LINEAR,borderValue=0)#command+鼠标可查看函数参数的意义，第二个参数为旋转仿射矩阵，第三个参数为变换后图像的尺寸，第四个参数为边界填充值
cv2.imshow('rotate',rotate)
cv2.imwrite("rotate.jpg",rotate)
rotate1=cv2.rotate(src,cv2.ROTATE_90_CLOCKWISE)#顺时针旋转90度
cv2.imshow('rotate1',rotate1)
'''
图像平移
'''
mat=np.float32([[1,0,100],[0,1,50]])
shift=cv2.warpAffine(src,mat,src.shape[:2],borderValue=0)
cv2.imshow('shift',shift)
'''
图像镜像
'''
flip=cv2.flip(src,1)#第二个参数为0表示绕x轴正翻转，参数为1表示绕轴正翻转与，参数为-1表示绕x轴、y轴正翻转
cv2.imshow('flip',flip)
"""
射影变换,求射影变换矩阵需要4个点,3*3矩阵的最后一个元素默认是1
"""
r,c=src.shape[:2]
set3=np.array([[0,0],[c-1,0],[0,r-1],[c-1,r-1]],dtype=np.float32)
set4=np.array([[50,50],[c/3,50],[50,r-1],[c-1,r-1]],dtype=np.float32)
mat_perspective=cv2.getPerspectiveTransform(set3,set4)#数据必须是32浮点型
print(mat_perspective)
perspective=cv2.warpPerspective(src,mat_perspective,(r,c))
cv2.imshow('Perspective',perspective)
'''
极坐标变换：可以校正图像中的圆形物体或被包围在圆环中的物体
cv2.cartToPolar(),cv2.polarToCart()
'''
'''
cv2.circle()：图像中画圆
cv2.rectangle()
cv2.ellipse()
cv2.line()
'''
linear_polar=cv2.linearPolar(src,(200,300),500,cv2.INTER_LINEAR)#线性极坐标函数
cv2.imshow('linear polar',linear_polar)
log_polar=cv2.logPolar(src,(200,300),100,cv2.WARP_FILL_OUTLIERS)
cv2.imshow('log polar',log_polar)




cv2.waitKey(0)
cv2.destroyAllWindows()