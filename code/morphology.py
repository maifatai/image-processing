import cv2
import numpy as np
'''
获取结构元
cv2.getStructuringElement(shape=,ksize=,anchor=)
shape:cv2.MORPH_RECT为矩形结构元，cv2.MORPH_ELLIPSE为椭圆结构元，
cv2.MORPH_CROSS为十字交叉结构元
ksize:结构元的尺寸
anchor：结构元的锚点
腐蚀：取结构元中像素值最小
cv2.erode(src=,kernel=,anchor=,iterations=,borderType=,borderValue=)
kernel：结构元
anchor：结构元的锚点
iterations：腐蚀操作的次数
borderType：边界扩充类型
borderValue：边界扩充值
膨胀
cv2.dilate(src, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None)
形态学操作的函数
cv2.morphologyEx(src, op, kernel, dst=None, anchor=None, iterations=None, borderType=None, borderValue=None)
op:形态学处理的各种运算,cv2.MORPH_OPEN为开运算（先腐蚀后膨胀），cv2.MORPH_CLOSE为闭运算
cv2.MORPH_ERODE为腐蚀，cv2.MORPH_DILATE为膨胀，cv2.MORPH_TOPHAT为顶帽操作(校正光照不均匀)，
cv2.MORPH_BLACKHAT为底帽运算，cv2.MORPH_GRADIENT为形态梯度
kernel：结构元
anchor：结构元的锚点
iterations：操作的次数
borderType：边界扩充类型
borderValue：边界扩充值
'''
print(cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE)
src=cv2.imread('lena.jpg',0)
radius,it=1,1#结构元半径和迭代次数
kernal_style,max_kernal=0,2
max_radius,max_it=20,20
morphology=[cv2.MORPH_ERODE,cv2.MORPH_DILATE,cv2.MORPH_OPEN,cv2.MORPH_CLOSE,
            cv2.MORPH_GRADIENT,cv2.MORPH_TOPHAT,cv2.MORPH_BLACKHAT]
kernal=[cv2.MORPH_RECT,cv2.MORPH_CROSS,cv2.MORPH_ELLIPSE]
def nothing(*args):
    pass
for i in morphology:
    cv2.namedWindow('morphology '+str(i),1)
    cv2.createTrackbar('radius','morphology '+str(i),radius,max_radius,nothing)#创建调节结构元半径的进度条
    cv2.createTrackbar('it','morphology '+str(i),it,max_it,nothing)#创建调节迭代次数的进度条
    cv2.createTrackbar('kernal_style', 'morphology ' + str(i), kernal_style, max_kernal, nothing)  # 创建调节结构元类型的进度条
    while True:
        radius=cv2.getTrackbarPos('radius','morphology '+str(i))#获取进度条上当前radius的值
        it=cv2.getTrackbarPos('it','morphology '+str(i))
        kernal_style= cv2.getTrackbarPos('kernal_style', 'morphology ' + str(i))
        s=cv2.getStructuringElement(kernal_style,(2*radius+1,2*radius+1))
        d=cv2.morphologyEx(src,i,s,iterations=it)
        cv2.imshow('morphology '+str(i),d)
        ch=cv2.waitKey(5)
        if ch==27:#按下ESC键退出内循环
            break
cv2.destroyAllWindows()