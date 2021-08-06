'''
full卷积、valid卷积、same卷积、边界填充方法

'''
import numpy as np
import cv2
from scipy import signal
'''
full卷积：矩阵M的大小为m*n,卷积核k的大小为p*q，full卷积的结果为(m+p-1)*(n+q-1),注意边界填充
valid卷积：矩阵M的大小为m*n,卷积核k的大小为p*q，full卷积的结果为(m-p+1)*(n-q+1)
same卷积：矩阵M的same卷积结果与原矩阵的大小一致，注意锚点的选择
边界填充方法
(1)在矩阵I边界外填充常数，通常进行的是0扩充。
(2)通过重复I边界处的行和列，对输人矩阵进行扩充，使卷积在边界处可计算。
(3)卷绕输人矩阵，即矩阵的平铺。
(4)以矩阵边界为中心，令矩阵外某位置上未定义的灰度值等于图像内其镜像位置的灰度值，
   这种处理方式会令结果产生最小程度的干扰
cv2.copyMakeBorder(src, top, bottom, left, right, borderType, dst=None, value=None)
    .   @param src Source image.
    .   @param dst Destination image of the same type as src and the size Size(src.cols+left+right,
    .   src.rows+top+bottom) .
    .   @param top the top pixels
    .   @param bottom the bottom pixels
    .   @param left the left pixels
    .   @param right Parameter specifying how many pixels in each direction from the source image rectangle
    .   to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs
    .   to be built.
    .   @param borderType Border type. 
        BORDER_REPLICATE:边界复制
        BORDER_CONSTANT:常数扩充
        BORDER_REFLECT:反射扩充
        BORDER_REFLECT_101:以边界为中心反射扩充
        BORDER_WRAP:平铺扩充
    .   @param value Border value if borderType==BORDER_CONSTANT 
    
    
    scipy.signal.convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0)
        Parameters
    ----------
    in1 : array_like
        First input.
    in2 : array_like
        Second input. Should have the same number of dimensions as `in1`.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output:

        ``full``
           The output is the full discrete linear convolution
           of the inputs. (Default)
        ``valid``
           The output consists only of those elements that do not
           rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
           must be at least as large as the other in every dimension.
        ``same``
           The output is the same size as `in1`, centered
           with respect to the 'full' output.
    boundary : str {'fill', 'wrap', 'symm'}, optional
        A flag indicating how to handle boundaries:

        ``fill``常数扩充
           pad input arrays with fillvalue. (default)
        ``wrap``反射扩充
           circular boundary conditions.
        ``symm``平铺扩充
           symmetrical boundary conditions.

    fillvalue : scalar, optional
        Value to fill pad input arrays with. Default is 0.

    Returns
    -------
    out : ndarray
        A 2-dimensional array containing a subset of the discrete linear
        convolution of `in1` with `in2`.
'''
mat=np.array([[5,1,7],[1,5,9],[2,6,2]])
replicate=cv2.copyMakeBorder(mat,2,2,2,2,borderType=cv2.BORDER_REPLICATE)
constant=cv2.copyMakeBorder(mat,2,2,2,2,borderType=cv2.BORDER_CONSTANT)
reflect=cv2.copyMakeBorder(mat,2,2,2,2,borderType=cv2.BORDER_REFLECT)
reflect_101=cv2.copyMakeBorder(mat,2,2,2,2,borderType=cv2.BORDER_REFLECT_101)
wrap=cv2.copyMakeBorder(mat,2,2,2,2,borderType=cv2.BORDER_WRAP)
print('replicate\n',replicate)
print('constant\n',constant)
print('reflect\n',reflect)
print('reflect_101\n',reflect_101)
print('wrap\n',wrap)
src=cv2.imread('lena.jpg',1)
replicate=cv2.copyMakeBorder(src,src.shape[0],src.shape[0],src.shape[0],src.shape[0],borderType=cv2.BORDER_REPLICATE)
constant=cv2.copyMakeBorder(src,src.shape[0],src.shape[0],src.shape[0],src.shape[0],borderType=cv2.BORDER_CONSTANT)
reflect=cv2.copyMakeBorder(src,src.shape[0],src.shape[0],src.shape[0],src.shape[0],borderType=cv2.BORDER_REFLECT)
reflect_101=cv2.copyMakeBorder(src,src.shape[0],src.shape[0],src.shape[0],src.shape[0],borderType=cv2.BORDER_REFLECT_101)
wrap=cv2.copyMakeBorder(src,src.shape[0],src.shape[0],src.shape[0],src.shape[0],borderType=cv2.BORDER_WRAP)
cv2.imshow('replicate',replicate)
cv2.imshow('constant',constant)
cv2.imshow('reflect',reflect)
cv2.imshow('reflect_101',reflect_101)
cv2.imshow('wrap',wrap)
#定义高斯滤波
def gaussblur(img,sigma,r,c,bound='fill',fillvalue=0):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kernal_x=cv2.getGaussianKernel(ksize=c,sigma=sigma,ktype=cv2.CV_64F)#获取高斯核
    kernal_x=np.transpose(kernal_x)
    gaussblur_x=signal.convolve2d(img,kernal_x,mode='same',boundary=bound,fillvalue=fillvalue)
    kernal_y=cv2.getGaussianKernel(ksize=r,sigma=sigma,ktype=cv2.CV_64F)#获取高斯核
    gaussblur_xy = signal.convolve2d(gaussblur_x, kernal_y, mode='same', boundary=bound, fillvalue=fillvalue)
    gaussblur_xy=np.round(gaussblur_xy).astype(np.uint8)
    return gaussblur_xy
dst=gaussblur(src,2,5,5,'symm')
cv2.imshow('gauss blur 5*5',dst)
gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
dst1=cv2.GaussianBlur(gray,ksize=(5,5),sigmaX=2,sigmaY=2,borderType=cv2.BORDER_REFLECT)
cv2.imshow('cv gauss blur 5*5',dst1)

#均值核
def mean_kernel(size):
    r,c=size[0],size[1]
    kernel_x=np.ones((r,1))
    kernel_y=np.ones((1,c))
    kernel=signal.convolve2d(kernel_x,kernel_y,mode='full')
    kernel=kernel/np.sum(kernel)
    return kernel
print(mean_kernel((5,5)))





cv2.waitKey(0)
cv2.destroyAllWindows()
