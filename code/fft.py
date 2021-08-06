'''
DFT、IDFT、FFT、IFFT
幅值谱、相角谱
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('lena.jpg',0)
fft=np.fft.fft2(img)
fft_abs=np.log(np.abs(fft))
fft_shift=np.fft.fftshift(fft)
fft_shift_abs=np.log(np.abs(fft_shift))

ifft_shift=np.fft.ifftshift(fft_shift)
ifft=np.fft.ifft2(fft)
ifft_abs=np.abs(ifft)

plt.subplot(221),plt.title('fft'),plt.imshow(fft_abs,cmap='gray')
plt.subplot(222),plt.title('ifft'),plt.imshow(ifft_abs,cmap='gray')
plt.subplot(223),plt.title('fft shift'),plt.imshow(fft_shift_abs,cmap='gray')
plt.subplot(224),plt.title('ifft shift'),plt.imshow(ifft_abs,cmap='gray')
plt.legend()
plt.show()

#opencv的傅里叶和逆傅里叶变换
'''
cv2.dft(src=,dst=,flags=)
src：只支持CV_32F和CV_64F的单通道或者双通道
flags：
cv2.DFT_COMPLEX_OUTPUT:输出复数形式
cv2.DFT_REAL_OUTPUT：只输出实部
cv2.DFT_INVERSE：傅里叶逆变换
cv2.DFT_SCALE：是否除以M*N
cv2.DFT_ROWS：输入矩阵的每行进行傅里叶变换或者逆变换
在进行傅里叶逆变换的时候，常用的组合是cv2.DFT_INVERSE+cv2.DFT_SCALE+cv2.DFT_COMPLEX_OUTPUT
'''
src=cv2.imread('lena.jpg')
def dft(image):
    img=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    dft_img=cv2.dft(np.float32(img),flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift=np.fft.fftshift(dft_img)
    ## cv2.magnitude()将实部和虚部转换为实部，乘以20将结果放大
    magnitude_spectrum=20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    plt.imshow(magnitude_spectrum,cmap='gray')
    plt.title('cv dft')
    plt.show()
    return dft_shift
def idft(image):
    dst_idft=np.fft.ifftshift(image)
    cv_idft=cv2.idft(dst_idft)
    if_magnitude=cv2.magnitude(cv_idft[:,:,0],cv_idft[:,:,1])
    plt.imshow(if_magnitude,cmap='gray')
    plt.title('cv idft')
    plt.show()
    return if_magnitude
dft1=dft(src)
idft(dft1)
#FFT
def FFT(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rpadded=cv2.getOptimalDFTSize(img.shape[0])#得到快速傅里叶变换的最优尺寸
    cpadded=cv2.getOptimalDFTSize(img.shape[1])
    fft=np.zeros((rpadded,cpadded),np.float32)#边缘扩充，下边缘和右边缘的扩充值为0
    fft[:img.shape[0],:img.shape[1]]=img
    dft_img = cv2.dft(np.float32(fft), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft_img)
    ## cv2.magnitude()计算两个矩阵对应位置平方和的平方根，用来计算幅度谱
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    #dft_shift[:, :, 0]表示傅里叶变换的实部， dft_shift[:, :, 1]表示傅里叶变换的虚部
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('FFT')
    plt.show()
    return dft_shift
def IFFT(src,size):
    src=np.fft.ifftshift(src)
    ifft=np.zeros(src.shape[:2],np.float32)
    ifft=cv2.dft(src,ifft,cv2.DFT_REAL_OUTPUT+cv2.DFT_INVERSE+cv2.DFT_SCALE)
    dst=np.copy(ifft[:size[0],:size[1]])
    plt.imshow(dst, cmap='gray')
    plt.title('IFFT')
    plt.show()
    return dst
def phasespectrum(fft2):
    phase=np.arctan2(fft2[:,:,1],fft2[:,:,0])#计算傅里叶变换的相位谱
    # dft_shift[:, :, 0]表示傅里叶变换的实部， dft_shift[:, :, 1]表示傅里叶变换的虚部
    spectrum=phase/np.pi*180
    plt.imshow(spectrum, cmap='gray')
    plt.title('phasespectrum')
    plt.show()
    return spectrum
phasespectrum(dft(src))
src=cv2.imread('lena.jpg')
fft1=FFT(src)
ifft1=IFFT(fft1,src.shape[:2])
fft_phase=phasespectrum(fft1)

#谱残差显著性检测
def spectralresidual(src):
    #step1:计算图像的快速傅里叶变换
    fft1=FFT(src)
    magnitude_spectrum = np.log(cv2.magnitude(fft1[:, :, 0], fft1[:, :, 1])+1)#幅度谱
    magnitude_spectrum=cv2.normalize(magnitude_spectrum,magnitude_spectrum,0,1,norm_type=cv2.NORM_MINMAX)
    phase=phasespectrum(fft1)
    #step2:计算相位谱，余弦谱，正弦谱
    sin=np.sin(phase)
    cos=np.cos(phase)
    #step3：对幅度谱进行均值滤波
    mean=cv2.boxFilter(magnitude_spectrum,cv2.CV_32FC1,(3,3))
    #step4:计算残差谱
    residual=magnitude_spectrum-mean
    #step5:谱残差的幂指数运算
    exp=np.exp(residual)
    #计算实部和虚部
    real=exp*cos
    imaginary=exp*sin
    #合并实部和虚部
    com=np.zeros((real.shape[0],real.shape[1],2),np.float32)
    com[:,:,0]=real
    com[:,:,1]=imaginary
    #step6:根据新的幅值谱和相位谱，进行傅里叶逆变换
    ifft1=np.zeros(com.shape,np.float32)
    cv2.dft(com,ifft1,cv2.DFT_COMPLEX_OUTPUT+cv2.DFT_INVERSE)
    #step7:显著性
    salien=np.power(ifft1[:,:,0],2)+np.power(ifft1[:,:,1],2)
    salien=cv2.GaussianBlur(salien,(11,11),2.5)
    cv2.normalize(salien,salien,0,1,cv2.NORM_MINMAX)
    salien=salien/np.max(salien)
    salien=np.power(salien,0.5)
    salien=np.round(salien*255).astype(np.uint8)
    plt.imshow(salien, cmap='gray')
    plt.title('spectralresidual')
    plt.show()
    return salien
dst=spectralresidual(src)





cv2.waitKey(0)
cv2.destroyAllWindows()






