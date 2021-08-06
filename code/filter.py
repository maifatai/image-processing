'''
均值滤波、中值滤波、最大值、最小值、中点滤波、高斯滤波、双边滤波、导向滤波
cv2.filter2D()函数是进行相关，不是进性卷积，必须要用cv2.flip(img,-1)的作用是使图像进行翻转
cv2.flip(filename, flipcode)
filename：需要操作的图像
flipcode：翻转方式
flipcode
1	水平翻转
0	垂直翻转
-1	水平垂直翻转
'''
import cv2
import numpy as np

img=cv2.imread('lena.jpg',0)
new=np.array(img)
#加椒盐噪声
noise=1000
for i in range(noise):
    x=np.random.randint(0,img.shape[1])
    y=np.random.randint(0,img.shape[0])
    new[y,x]=255
    x=np.random.randint(0,img.shape[1])
    y=np.random.randint(0,img.shape[0])
    new[y,x]=0
result=cv2.blur(new,(3,3))#均值滤波
res=np.hstack((new,result))
cv2.imshow('blur',res)

k=np.ones((3,3),np.float32)/9
dst=cv2.filter2D(new,-1,k)
result1=np.hstack((new,dst))
print(np.all(result==result1))
cv2.imshow('blur1',result1)

#中值滤波
medianblur=cv2.medianBlur(new,3)
result2=np.hstack((new,medianblur))
cv2.imshow('mediumblur',result2)
#最大值滤波器，最小值滤波器,中点滤波
max_img=np.array(new)
min_img=np.array(new)
medium_img=np.array(new)
for i in range(1,new.shape[0]-2):
    for j in range(1,new.shape[1]-2):
        temp=img[i-1:i+2,j-1:j+2]#8领域
        max_img[i,j]=np.max(temp)
        min_img[i,j]=np.min(temp)
        medium_img[i,j]=int(np.min(temp)/2+np.max(temp)/2)
result3=np.hstack((new,max_img))
cv2.imshow('maxblur',result3)
result4=np.hstack((new,min_img))
cv2.imshow('minblur',result4)
result5=np.hstack((new,medium_img))
cv2.imshow('mediumblur',result5)

#高斯滤波
'''
cv2.GaussianBlur（src,ksize,sigmaX [,DST [,sigmaY [,borderType ] ] ] ）
src：要进行滤波的原图像；
ksize: 高斯核的大小，取值一般为奇数，如(3,3)；
sigmaX和sigmaY是高斯标准差，一般有了高斯核ksize的大小的话，这两个参数可以省；
borderType：像素外推方法。
'''
gauss=cv2.GaussianBlur(new,(3,3),0)
result6=np.hstack((new,gauss))
cv2.imshow('gauss filter',result6)

#双边滤波
"""
双边滤波是一种非线性的滤波方法，能够保持边界清晰的情况下有效的去除噪声，它拥有类似相机里美颜的效果。
双边滤波之所以能够做到保边去噪的效果，是由于它的有两个核：空间域核和值域核，比高斯滤波只有一个值域核多了一个。
空间域核是由像素位置欧式距离决定的模板权值。
值域核是由像素值的差值决定的模板权值。
cv2.bilateralFilter(src=img,d=13,sigmaColor=75,sigmaSpace=75)#参数名可省。
  src：原图像；
  d：像素的邻域直径；
  sigmaColor：颜色空间的标准方差，一般尽可能大；
  sigmaSpace：坐标空间的标准方差(像素单位)，一般尽可能小。
"""
img1=cv2.imread('lena.jpg')
bilateral=cv2.bilateralFilter(img1,13,75,75)
result7=np.hstack((img1,bilateral))
cv2.imshow('bilateral Filter',result7)

'''
导向滤波：在平滑图像的基础上，有良好的保边作用，在细节增强等方面有良好的表现
'''
def guidefilter(guide_img,src,radius,eps):
    guide_img=cv2.normalize(guide_img,guide_img,1,0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
    src = cv2.normalize(src, src, 1, 0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    r,c=guide_img.shape
    mean_guide_img=cv2.blur(guide_img,(radius,radius),borderType=cv2.BORDER_DEFAULT)
    mean_src=cv2.blur(src,(radius,radius),borderType=cv2.BORDER_DEFAULT)
    mean_src_guide=cv2.blur(guide_img*src,(radius,radius),borderType=cv2.BORDER_DEFAULT)
    mean_src_mul=cv2.blur(src*src,(radius,radius),borderType=cv2.BORDER_DEFAULT)
    var_src=mean_src_mul-mean_src*mean_src
    cov_src_guide=mean_src_guide-mean_src*mean_guide_img
    a=cov_src_guide/(var_src+eps)
    b=mean_src_guide-a*mean_src
    mean_a=cv2.blur(a,(radius,radius),borderType=cv2.BORDER_DEFAULT)
    mean_b = cv2.blur(b, (radius, radius), borderType=cv2.BORDER_DEFAULT)
    q=mean_a*guide_img-mean_b
    q=cv2.normalize(q,q,255,0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    return q
dst1=guidefilter(img,img,5,0.04)
dst1=np.power(dst1,0.6)
dst1=cv2.normalize(dst1,dst1,255,0,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
cv2.imshow('guidefilter',dst1)



cv2.waitKey(0)
cv2.destroyAllWindows()