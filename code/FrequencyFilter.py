'''
理想高\低滤波、巴特沃斯、高斯
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
class ILF:
    '''
    img:input image
    d:the diameter of filter
    '''
    def __init__(self,img,d):
        self._img=img
        self._d=d
        self.row,self.col=img.shape
    def kernel(self):
        r,c=self._img.shape
        u,v=np.arange(r),np.arange(c)
        u,v=np.meshgrid(u,v)
        low_pass=np.sqrt((u-r/2)**2+(v-c/2)**2)#频谱移到中心,距离中心的距离矩阵
        low_pass[low_pass<self._d]=1
        low_pass[low_pass>=self._d]=0
        return low_pass
    def ideal_low_pass(self):
        k=self.kernel()
        gray=self._img.copy()
        dft=np.fft.fft2(np.float64(gray))
        gray_shift=np.fft.fftshift(dft)
        dst_filter=k*gray_shift#低通滤波器与dft的图像相乘
        dst_ishift=np.fft.ifftshift(dst_filter)
        dst_ifft=np.fft.ifft2(dst_ishift)
        dst_ifft=np.abs(dst_ifft)
        return dst_ifft
    def plot(self,string):
        plt.title(string)
        dst=self.ideal_low_pass()
        plt.imshow(dst,cmap='gray')
        plt.show()
class IHF:
    '''
    img:input image
    d:the diameter of filter
    '''
    def __init__(self,img,d):
        self._img=img
        self._d=d
        self.row,self.col=img.shape
    def kernel(self):
        r,c=self._img.shape
        u,v=np.arange(r),np.arange(c)
        u,v=np.meshgrid(u,v)
        high_pass=np.sqrt((u-r/2)**2+(v-c/2)**2)#频谱移到中心,距离中心的距离矩阵
        high_pass[high_pass<self._d]=0
        high_pass[high_pass>=self._d]=1
        return high_pass
    def ideal_high_pass(self):
        k=self.kernel()
        gray=self._img.copy()
        dft=np.fft.fft2(np.float64(gray))
        gray_shift=np.fft.fftshift(dft)

        dst_filter=k*gray_shift#低通滤波器与dft的图像相乘
        dst_ishift=np.fft.ifftshift(dst_filter)
        dst_ifft=np.fft.ifft2(dst_ishift)
        dst_ifft=np.abs(dst_ifft)
        return dst_ifft
    def plot(self,string):
        plt.title(string)
        dst=self.ideal_high_pass()
        plt.imshow(dst,cmap='gray')
        plt.show()
class BLF:
    '''
    img:input image
    d:the diameter of filter
    '''
    def __init__(self,img,d,n):
        self._img=img
        self._d=d
        self.row,self.col=img.shape
        self._n=n
    def kernel(self):
        r,c=self._img.shape
        u,v=np.arange(r),np.arange(c)
        u,v=np.meshgrid(u,v)
        low_pass=np.sqrt((u-r/2)**2+(v-c/2)**2)#频谱移到中心,距离中心的距离矩阵
        low_pass=1/(1+low_pass/self._d)**(2*self._n)
        return low_pass
    def butterworth_low_pass(self):
        k=self.kernel()
        gray=self._img.copy()
        dft=np.fft.fft2(np.float64(gray))
        gray_shift=np.fft.fftshift(dft)
        dst=np.zeros_like(gray_shift)
        dst_filter=k*gray_shift#低通滤波器与dft的图像相乘
        dst_ishift=np.fft.ifftshift(dst_filter)
        dst_ifft=np.fft.ifft2(dst_ishift)
        dst_ifft=np.abs(dst_ifft)
        return dst_ifft
    def plot(self,string):
        plt.title(string)
        dst=self.butterworth_low_pass()
        plt.imshow(dst,cmap='gray')
        plt.show()
class BHF:
    '''
    img:input image
    d:the diameter of filter
    '''
    def __init__(self,img,d,n):
        self._img=img
        self._d=d
        self.row,self.col=img.shape
        self._n=n
    def kernel(self):
        r,c=self._img.shape
        u,v=np.arange(r),np.arange(c)
        u,v=np.meshgrid(u,v)
        high_pass=np.sqrt((u-r/2)**2+(v-c/2)**2)#频谱移到中心,距离中心的距离矩阵
        high_pass=1/(1+self._d/high_pass)**(2*self._n)
        return high_pass
    def butterworth_high_pass(self):
        k=self.kernel()
        gray=self._img.copy()
        dft=np.fft.fft2(np.float64(gray))
        gray_shift=np.fft.fftshift(dft)
        dst_filter=k*gray_shift#低通滤波器与dft的图像相乘
        dst_ishift=np.fft.ifftshift(dst_filter)
        dst_ifft=np.fft.ifft2(dst_ishift)
        dst_ifft=np.abs(dst_ifft)
        return dst_ifft
    def plot(self,string):
        plt.title(string)
        dst=self.butterworth_high_pass()
        plt.imshow(dst,cmap='gray')
        plt.show()
class GLF:
    '''
    img:input image
    d:the diameter of filter
    '''
    def __init__(self,img,d):
        self._img=img
        self._d=d
        self.row,self.col=img.shape
    def kernel(self):
        r,c=self._img.shape
        u,v=np.arange(r),np.arange(c)
        u,v=np.meshgrid(u,v)
        low_pass=np.sqrt((u-r/2)**2+(v-c/2)**2)#频谱移到中心,距离中心的距离矩阵
        low_pass=np.exp(-(low_pass**2)/(2*self._d**2))
        return low_pass
    def gauss_low_pass(self):
        k=self.kernel()
        gray=self._img.copy()
        dft=np.fft.fft2(np.float64(gray))
        gray_shift=np.fft.fftshift(dft)
        dst_filter=k*gray_shift#低通滤波器与dft的图像相乘
        dst_ishift=np.fft.ifftshift(dst_filter)
        dst_ifft=np.fft.ifft2(dst_ishift)
        dst_ifft=np.abs(dst_ifft)
        return dst_ifft
    def plot(self,string):
        plt.title(string)
        dst=self.gauss_low_pass()
        plt.imshow(dst,cmap='gray')
        plt.show()
class GHF:
    '''
    img:input image
    d:the diameter of filter
    '''
    def __init__(self,img,d):
        self._img=img
        self._d=d
        self.row,self.col=img.shape
    def kernel(self):
        r,c=self._img.shape
        u,v=np.arange(r),np.arange(c)
        u,v=np.meshgrid(u,v)
        high_pass=np.sqrt((u-r/2)**2+(v-c/2)**2)#频谱移到中心,距离中心的距离矩阵
        high_pass=1-np.exp(-(high_pass**2)/(2*self._d**2))
        return high_pass
    def gauss_high_pass(self):
        k=self.kernel()
        gray=self._img.copy()
        dft=np.fft.fft2(np.float64(gray))
        gray_shift=np.fft.fftshift(dft)
        dst_filter=k*gray_shift#低通滤波器与dft的图像相乘
        dst_ishift=np.fft.ifftshift(dst_filter)
        dst_ifft=np.fft.ifft2(dst_ishift)
        dst_ifft=np.abs(dst_ifft)
        return dst_ifft
    def plot(self,string):
        plt.title(string)
        dst=self.gauss_high_pass()
        plt.show()
#同态滤波
def homomorphic_filter(src, d0=10, r1=0.5, rh=2, c=4, h=2.0, l=0.5):
    gray = src.copy()
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)
    rows, cols = gray.shape
    gray_fft = np.fft.fft2(gray)
    gray_fftshift = np.fft.fftshift(gray_fft)
    dst_fftshift = np.zeros_like(gray_fftshift)
    M, N = np.meshgrid(np.arange(-cols // 2, cols // 2), np.arange(-rows//2, rows//2))
    D = np.sqrt(M ** 2 + N ** 2)
    Z = (rh - r1) * (1 - np.exp(-c * (D ** 2 / d0 ** 2))) + r1
    dst_fftshift = Z * gray_fftshift
    dst_fftshift = (h - l) * dst_fftshift + l
    dst_ifftshift = np.fft.ifftshift(dst_fftshift)
    dst_ifft = np.fft.ifft2(dst_ifftshift)
    dst = np.real(dst_ifft)
    dst = np.uint8(np.clip(dst, 0, 255))
    plt.imshow(dst,cmap='gray')
    plt.title('homomorphic filter')
    plt.show()
    return dst


if __name__=="__main__":
    src=cv2.imread('lena.jpg',0)
    ILF(src, 5).plot('ILFP 5')
    ILF(src, 10).plot('ILFP 10')
    ILF(src, 20).plot('ILFP 20')
    ILF(src, 40).plot('ILFP 40')
    ILF(src, 80).plot('ILFP 80')
    ILF(src, 100).plot('ILFP 100')
    #
    # IHF(src, 5).plot('IHFP 5')
    # IHF(src, 10).plot('IHFP 10')
    # IHF(src, 20).plot('IHFP 20')
    # IHF(src, 40).plot('IHFP 40')
    # IHF(src, 80).plot('IHFP 80')
    # IHF(src, 100).plot('IHFP 100')
    #
    # BLF(src, 5, 1).plot('BLP 5 1')
    # BLF(src, 5, 2).plot('BLP 5 2')
    # BLF(src, 5, 3).plot('BLP 5 3')
    # BLF(src, 5, 4).plot('BLP 5 4')
    # BLF(src, 20, 1).plot('BLP 20 1')
    # BLF(src, 20, 2).plot('BLP 20 2')
    # BLF(src, 20, 3).plot('BLP 20 3')
    # BLF(src, 20, 4).plot('BLP 20 4')
    # BLF(src, 40, 1).plot('BLP 40 1')
    # BLF(src, 40, 2).plot('BLP 40 2')
    # BLF(src, 40, 3).plot('BLP 40 3')
    # BLF(src, 40, 4).plot('BLP 40 4')
    # BLF(src, 80, 1).plot('BLP 80 1')
    # BLF(src, 80, 2).plot('BLP 80 2')
    # BLF(src, 80, 3).plot('BLP 80 3')
    # BLF(src, 80, 4).plot('BLP 80 4')
    # BHF(src, 5, 1).plot('BLP 5 1')
    # BHF(src, 5, 2).plot('BLP 5 2')
    # BHF(src, 5, 3).plot('BLP 5 3')
    # BHF(src, 5, 4).plot('BLP 5 4')
    # BHF(src, 20, 1).plot('BLP 20 1')
    # BHF(src, 20, 2).plot('BLP 20 2')
    # BHF(src, 20, 3).plot('BLP 20 3')
    # BHF(src, 20, 4).plot('BLP 20 4')
    # BHF(src, 40, 1).plot('BLP 40 1')
    # BHF(src, 40, 2).plot('BLP 40 2')
    # BHF(src, 40, 3).plot('BLP 40 3')
    # BHF(src, 40, 4).plot('BLP 40 4')
    # BHF(src, 80, 1).plot('BLP 80 1')
    # BHF(src, 80, 2).plot('BLP 80 2')
    # BHF(src, 80, 3).plot('BLP 80 3')
    # BHF(src, 80, 4).plot('BLP 80 4')

    # GLF(src, 5).plot('GLFP 5')
    # GLF(src, 10).plot('GLFP 10')
    # GLF(src, 20).plot('GLFP 20')
    # GLF(src, 40).plot('GLFP 40')
    # GLF(src, 80).plot('GLFP 80')
    # GLF(src, 100).plot('GLFP 100')
    # GHF(src, 5).plot('GHFP 5')
    # GHF(src, 10).plot('GHFP 10')
    # GHF(src, 20).plot('GHFP 20')
    # GHF(src, 40).plot('GHFP 40')
    # GHF(src, 80).plot('GHFP 80')
    # GHF(src, 100).plot('GHFP 100')
    # homomorphic_filter(src)








