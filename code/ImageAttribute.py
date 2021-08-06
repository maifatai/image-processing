import cv2
'''
通道数及其数据类型，可以设置为CV_8UC(n)、CV_8SC(n)、CV_16SC(n)、CV_16UC(n)、
CV_32SC(n)、CV_32FC(n)、CV_ 64FC(n)，其中8U、8S、16S、16U、32S、32F、64F
前面的数字代表Mat中每一个数值所占的bit数，而lbyte=8bit,所以，32F就是占4字节的float类型，
64F是占8字节的doule类型，32S是占4字节的int类型，8U是占1字节的uchar类型，其他的类似;C(n)代表通道数，
当n=1时，即构造单通道矩阵或称二维矩阵，当n>1时，即构造多通道矩阵即三维矩阵，直观上就是n个二维矩阵组成的三维矩阵
'''
src=cv2.imread('lena.jpg',1)
print(src.shape)
print(src.size)#图像的总大小
print(src.dtype)#元素的类型
print(src.ndim)#秩，图像的维度
print(src.itemsize)#每个元素的大小，以字节为单位
print(type(src))
print(src[:,:,0])
