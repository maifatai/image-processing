'''
直方图、直方图均衡化
灰度直方图均衡
彩色直方图均衡
自适应直方图均衡化
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
src=cv2.imread('lena.jpg',0)
'''
matplotlib.pyplot.hist(x, bins=None, range=None, density=False, weights=None, 
cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, *, data=None, **kwargs)[source]
x	数据	数值类型
bins	条形数	int
color	颜色	"r","g","y","c"
density	是否以密度的形式显示	bool
range	x轴的范围	数值元组（起，终）
bottom	y轴的起始位置	数值类型
histtype	线条的类型	"bar":方形，"barstacked":柱形,
"step":"未填充线条"
"stepfilled":"填充线条"
align	对齐方式	"left":左，"mid":中间，"right":右
orientation	orientation	"horizontal":水平，"vertical":垂直
log	单位是否以科学计术法	bool
'''
plt.figure('histogram')
plt.title('plt:histogram of lena')
plt.hist(src.ravel(),256,range=(0,255),histtype='barstacked')
plt.show()
'''
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist

images:输入的图像
channels:选择图像的通道
mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
histSize:使用多少个bin(柱子)，一般为256
ranges:像素值的范围，一般为[0,255]表示0~255
'''
hist=cv2.calcHist([src],[0],None,[256],[0,256])
plt.title('cv:histogram of lena')
plt.plot(hist)
plt.show()

"""
彩色直方图
"""
def color_hist(image):
    color=('b','g','r')
    for i,color in enumerate(color):
        hist=cv2.calcHist([image],[i],None,[256],[0,255])
        plt.plot(hist,color)
        plt.xlim([0,255])
    #plt.show()
image=cv2.imread('lena.jpg',1)
plt.title('color hist')
color_hist(image)
plt.show()
'''
直方图均衡
灰度直方图均衡
彩色直方图均衡
'''
equ=cv2.equalizeHist(src)#直方图均衡
plt.figure(figsize=(8,8))
plt.suptitle('gray equalizeHist')
plt.subplot(221),plt.title('origin'),plt.imshow(src,'gray')
plt.subplot(222),plt.title('hist'),plt.hist(src.ravel(),256)
plt.subplot(223),plt.title('equalize'),plt.imshow(equ,'gray')
plt.subplot(224),plt.hist(equ.ravel(),256,[0,256]),plt.title('equalize hist')
plt.show()

b,g,r=cv2.split(image)
bh,gh,rh=cv2.equalizeHist(b),cv2.equalizeHist(g),cv2.equalizeHist(r)
result=cv2.merge((bh,gh,rh))
plt.figure(figsize=(8,8))
plt.suptitle('color equalizeHist')
plt.subplot(221),plt.title('origin'),plt.imshow(image[:,:,[2,1,0]])
plt.subplot(222),plt.title('hist'),color_hist(image)
plt.subplot(223),plt.title('equalize'),plt.imshow(result[:,:,[2,1,0]])
plt.subplot(224),plt.title('equalize hist'),color_hist(result)
plt.show()

'''
在YUV颜色空间，仅对亮度进行直方图均衡
'''
img_yuv=cv2.cvtColor(image,cv2.COLOR_BGR2YUV)
img_yuv[:,:,0]=cv2.equalizeHist(img_yuv[:,:,0])#仅对亮度进行直方图均衡
result=cv2.cvtColor(img_yuv,cv2.COLOR_YUV2RGB)#转换回彩色图像
plt.figure(figsize=(8,8))
plt.suptitle('YUV')
plt.subplot(221),plt.title('origin'),plt.imshow(image[:,:,[2,1,0]])
plt.subplot(222),plt.title('hist'),color_hist(image)
plt.subplot(223),plt.title('equalize'),plt.imshow(result[:,:,[2,1,0]])
plt.subplot(224),plt.title('YUV equalize hist'),color_hist(result)
plt.show()

"""
在HSV颜色空间，仅对亮度进行直方图均衡
"""
img_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
img_hsv[:,:,0]=cv2.equalizeHist(img_hsv[:,:,0])#仅对亮度进行直方图均衡
result=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)#转换回彩色图像
plt.figure(figsize=(8,8))
plt.suptitle('HSV')
plt.subplot(221),plt.title('origin'),plt.imshow(image[:,:,[2,1,0]])
plt.subplot(222),plt.title('hist'),color_hist(image)
plt.subplot(223),plt.title('equalize'),plt.imshow(result[:,:,[2,1,0]])
plt.subplot(224),plt.title('HSV equalize hist'),color_hist(result)
plt.show()

'''自适应直方图均衡化：将图像划分为不重叠的区域块，然后对每一块分别进行直方图均衡化'''
clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))#创建CLAHE对象
#clipLimit颜色对比度阈值，titleGridSize进行像素均衡化的网格大小
cll=clahe.apply(src)#限制对比度的自适应阈值均衡化
res=np.hstack((src,cll))
cv2.imshow('hist CLAHE',res)
plt.subplot(211),plt.title('origin'),plt.hist(src.ravel(),256,[0,256])
plt.subplot(212),plt.title('hist CLAHE'),plt.hist(res.ravel(),256,[0,256])
plt.show()

#直方图
def cal_hist(img,channel,histsize,rang):
    """
    :param img: 输入图像
    :param channel: 图像的通道
    :param histsize: 直方图中的条数
    :param rang: 范围
    :return:
    """
    src=img[:,:,channel]
    r,c=src.shape
    gray_hist=np.zeros([histsize],np.uint64)
    for i in range(r):
        for j in range(c):
            for k in range(histsize):
                if k*rang/histsize<=src[i][j]<(k+1)*rang/histsize:
                    gray_hist[k]+=1
    plt.hist(gray_hist,histsize,[0,256])
    plt.xlabel('level')
    plt.ylabel('number of pixels')
    plt.show()
cal_hist(image,0,10,256)

cv2.waitKey(0)
cv2.destroyAllWindows()