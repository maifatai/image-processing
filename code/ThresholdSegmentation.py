import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
阈值分割：OTSU、TRIANGER、熵算法、自适应阈值分割
二值图像的与、或、非、异或运算
'''
src=cv2.imread('lena.jpg',0)
plt.figure('histogram')
plt.title('plt：histogram of lena')
plt.hist(src.ravel(),256)
plt.show()
'''
全局阈值分割
'''

ret,binary_img=cv2.threshold(src,127,255,cv2.THRESH_BINARY)#必须有两个返回值，ret为阈值
cv2.imshow('binary',binary_img)
plt.imshow(binary_img,cmap='gray')
plt.title('binary')
plt.show()

'''
OTSU阈值处理:前景和背景差异的方差最大
'''
thres,otsu=cv2.threshold(src,0,255,cv2.THRESH_OTSU)
print(thres)
plt.imshow(otsu,cmap='gray')
plt.title('otsu')
plt.show()

'''
triangle 阈值处理
'''
tri_thres,triangle=cv2.threshold(src,0,255,cv2.THRESH_TRIANGLE)
print(tri_thres)
plt.imshow(triangle,cmap='gray')
plt.title('triangle')
plt.show()

tri_thres1,triangle1=cv2.threshold(src,0,255,cv2.THRESH_TRIANGLE+cv2.THRESH_BINARY_INV)
print(tri_thres1)
plt.imshow(triangle1,cmap='gray')
plt.title('triangle binary inv')
plt.show()

'''
熵算法
'''
def threshold_entropy(img):
    r,c=img.shape
    hist=cv2.calcHist([src],[0],None,[256],[0,256])
    normalize=(hist/float(r*c)).ravel()
    #计算累加直方图
    zeromoment=np.zeros([256],np.float32)
    for i in range(256):
        if i==0:
            zeromoment[i]=normalize[i]
        else:
            zeromoment[i]=zeromoment[i-1]+normalize[i]
    #计算各个灰度级的熵
    entropy=np.zeros([256],np.float32)
    for i in range(256):
        if i==0:
            if normalize[i]==0:
                entropy[i]=0
            else:
                entropy[i]=-normalize[i]*np.log10(normalize[i])
        else:
            if normalize[i]==0:
                entropy[i]=entropy[i-1]
            else:
                entropy[i]=entropy[i-1]--normalize[i]*np.log10(normalize[i])
    #找阈值
    ft=np.zeros([256],np.float32)
    f1,f2=0,0
    totalentropy=entropy[255]
    for i in range(255):
        max_front=np.max(normalize[0:i+1])
        max_back=np.max(normalize[i+1:256])
        if(max_front==0 or zeromoment[i]==0 or max_front==1 or zeromoment[i]==1 or totalentropy==0):
            f1=0
        else:
            f1=entropy[i]/totalentropy*(np.log10(zeromoment[i])/np.log10(max_front))
        if(max_back==0 or 1-zeromoment[i]==0 or max_back==1 or 1-zeromoment[i]==1):
            f2=0
        else:
            f2=(1-entropy[i]/totalentropy)*(np.log10(1-zeromoment[i])/np.log10(max_back))
        ft[i]=f1+f2
    #找到最大值的索引
    thresh_index=np.where(ft==np.max(ft))#np.where获取下标
    thresh=thresh_index[0][0]
    #阈值处理
    dst=np.copy(img)
    dst[dst>thresh]=255#形成布尔数组下标
    dst[dst<=thresh]=0
    plt.imshow(dst,cmap='gray')
    plt.title('entropy threshold')
    plt.show()
    return dst
threshold_entropy(src)

'''
局部阈值分割：针对输入矩阵的每个位置的值都有相应的阈值，这些阈值构成了和输入矩阵同尺寸的矩阵thresh，
其核心是计算阈值矩阵。
自适应阈值分割：利用平滑处理后的图像作为阈值矩阵，平滑算子的尺寸决定了分割出来的物体的尺寸。
自适应阈值分割可以克服光照不均匀的影响
'''
adapt_mean=cv2.adaptiveThreshold(src,255,adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=9,C=0.15)
plt.imshow(adapt_mean,cmap='gray')
plt.title('adaptive threshold mean')
plt.show()
adapt_gauss=cv2.adaptiveThreshold(src,255,adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=9,C=0.15)
plt.imshow(adapt_gauss,cmap='gray')
plt.title('adaptive threshold guass')
plt.show()

'''
二值图像的与、或、非、异或运算
'''
dst_and=cv2.bitwise_and(adapt_mean,adapt_gauss)
plt.imshow(dst_and,cmap='gray')
plt.title('binary and')
plt.show()
dst_or=cv2.bitwise_or(adapt_mean,adapt_gauss)
plt.imshow(dst_or,cmap='gray')
plt.title('binary or')
plt.show()
dst_not=cv2.bitwise_not(adapt_mean)
plt.imshow(dst_not,cmap='gray')
plt.title('binary not')
plt.show()
dst_xor=cv2.bitwise_xor(adapt_mean,adapt_gauss)
plt.imshow(dst_xor,cmap='gray')
plt.title('binary xor')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
