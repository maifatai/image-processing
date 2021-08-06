'''
Roberts算子、Prewitt、sobel、scharr算子、laplacian算子
#Kirsch算子、Robinson算子、canny算子、LOG、DOG、Marr-Hildreth
'''
import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
img=cv2.imread('pic/lena.jpg',0)

#Roberts算子
roberts_x=np.array([[-1,0],[0,1]],dtype=np.int)
roberts_y=np.array([[0,-1],[1,0]],dtype=np.int)
roberts_img_x=cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_16S,roberts_x))#数据转换为uint8.对于大于255的自动截断
roberts_img_y=cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_16S,roberts_y))
res=np.hstack((img,roberts_img_x))
cv2.imshow('roberts img x',res)
res1=np.hstack((img,roberts_img_y))
cv2.imshow('roberts img y',res1)

roberts=cv2.addWeighted(roberts_img_x,0.5,roberts_img_y,0.5,0)
res2=np.hstack((img,roberts))
cv2.imshow('roberts img',res2)
#Prewitt
prewitt_x=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
prewitt_y=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
prewitt_img_x=cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_16S,prewitt_x))#数据转换为uint8.对于大于255的自动截断
prewitt_img_y=cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_16S,prewitt_y))
prewitt=cv2.addWeighted(prewitt_img_x,0.5,prewitt_img_y,0.5,0)
res3=np.hstack((img,prewitt_img_x))
cv2.imshow('prewitt img x',res3)
res4=np.hstack((img,prewitt_img_y))
cv2.imshow('prewitt img x',res4)
res5=np.hstack((img,prewitt))
cv2.imshow('prewitt img x',res5)
#sobel
sobel_x=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
sobel_y=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
sobel_img_x=cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_16S,sobel_x))#数据转换为uint8.对于大于255的自动截断
sobel_img_y=cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_16S,sobel_y))
sobel=cv2.addWeighted(sobel_img_x,0.5,sobel_img_y,0.5,0)
res6=np.hstack((img,sobel_img_x))
cv2.imshow('sobel img x',res6)
res7=np.hstack((img,sobel_img_y))
cv2.imshow('sobel img x',res7)
res8=np.hstack((img,sobel))
cv2.imshow('sobel img x',res8)
#opencv中自带的函数
'''
cv2.Sobel(src=,ddepth=,dx=,dy=[,ksize=[,scale=[,delta=[,borderType]]]]])
ddepth:目标图像的深度，输出图像的数据类型，针对不同的输入图像，输出目标有不同的深度，-1表示与原图像相同的深度
dx，dy：表示求导的阶数，0表示不求导，一般为0，1，2
ksize:算子的大小
scale：缩放导数的比例常数，
delta：平移系数，所谓的步长
'''
sobel_x_cv=cv2.Sobel(img,cv2.CV_16S,1,0)
sobel_x_cv_img=cv2.convertScaleAbs(sobel_x_cv)
cv2.imshow('soble x cv',sobel_x_cv_img)
sobel_y_cv=cv2.Sobel(img,cv2.CV_16S,0,1)
sobel_y_cv_img=cv2.convertScaleAbs(sobel_y_cv)
cv2.imshow('soble x cv',sobel_y_cv_img)
sobel_cv=cv2.addWeighted(sobel_x_cv_img,0.5,sobel_y_cv_img,0.5,0)
cv2.imshow('sobel cv',sobel_cv)
#scharr算子
"""
cv2.Scharr(src, ddepth, dx, dy, dst=None, scale=None, delta=None, borderType=None):
参数和sobel算子的相同
scharr_x=np.array([3,0,-3],[10,0,-10],[3,0,-3])
scharr_y=np.array([3,10,3],[0,0,0],[-3,-10,-3])
scharr_45=np.array([0,3,10],[-3,0,3],[-10,-3,0])
scharr_135=np.array([10,3,0],[3,0,-3],[0,-3,-10])
"""
scharr_x_cv=cv2.Scharr(img,cv2.CV_16S,1,0)
scharr_x_cv_img=cv2.convertScaleAbs(scharr_x_cv)
cv2.imshow('scharrx cv',sobel_x_cv_img)
scharr_y_cv=cv2.Scharr(img,cv2.CV_16S,0,1)
scharr_y_cv_img=cv2.convertScaleAbs(scharr_y_cv)
cv2.imshow('scharr x cv',scharr_y_cv_img)
scharr_cv=cv2.addWeighted(scharr_x_cv_img,0.5,scharr_y_cv_img,0.5,0)
cv2.imshow('scharr cv',scharr_cv)
#laplacian算子
laplacian=cv2.Laplacian(img,cv2.CV_32F)
laplacian_img=cv2.convertScaleAbs(laplacian)
cv2.imshow('Laplacian',laplacian_img)
#Kirsch算子
def kirsch(img,boundfill='fill',fill_value=0):
    list_edge=[]#存储8个方向的边缘强度
    #第一步：8个边缘卷积算子分别和图像进行卷积，然后分别取绝对值得到边缘强度
    k1=np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]])
    k2=np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]])
    k3=np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]])
    k4=np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]])
    k5=np.array([[-3,-3,5],[-3,0,-5],[-3,-3,5]])
    k6=np.array([5,-3,-3,5,0,-3,5,-3,-3]).reshape(3,-1)
    k7=np.array([-3,-3,-3,-3,0,5,-3,5,5]).reshape(3,-1)
    k8=np.array([5,5,-3,5,0,-3,-3,-3,-3]).reshape(3,-1)
    kernal=[k1,k2,k3,k4,k5,k6,k7,k8]
    for i in kernal:
        img_ki=signal.convolve2d(img,i,mode='same',boundary=boundfill,fillvalue=fill_value)
        list_edge.append(np.abs(img_ki))
    #第二步：对上述的8个方向上的边缘强度，对应位置取最大值作为最后的边缘强度
    edge=list_edge[0]
    for i in range(len(list_edge)):
        edge=edge*(edge>=list_edge[i])+list_edge[i]*(edge<list_edge[i])
        '''上面的算法用于求最大值，很高级'''
    edge[edge/5>255]=255#阈值分割
    edge=edge.astype(np.uint8)
    plt.imshow(edge,cmap='gray')
    plt.title('Kirsch')
    plt.show()
    return edge
edge=kirsch(img,boundfill='symm')
#简单的素描结果，对上面的结果反色处理
pencil=255-edge
plt.imshow(pencil, cmap='gray')
plt.title('pencil')
plt.show()
#Robinson算子:与Kiesch相似
def robinson(img,boundfill='fill',fill_value=0):
    list_edge=[]#存储8个方向的边缘强度
    #第一步：8个边缘卷积算子分别和图像进行卷积，然后分别取绝对值得到边缘强度
    k1=np.array([1,1,1,1,-2,1,-1,-1,-1]).reshape(3,-1)
    k2=np.array([1,1,1,-1,-2,1,-1,-1,1]).reshape(3,-1)
    k3=np.array([-1,1,1,-1,-2,1,-1,1,1]).reshape(3,-1)
    k4=np.array([-1,-1,1,-1,-2,1,1,1,1]).reshape(3,-1)
    k5=np.array([-1,-1,-1,1,-2,1,1,1,1]).reshape(3,-1)
    k6=np.array([1,-1,-1,1,-2,-1,1,1,1]).reshape(3,-1)
    k7=np.array([1,1,-1,1,-2,-1,1,1,-1]).reshape(3,-1)
    k8=np.array([1,1,1,1,-2,-1,1,-1,-1]).reshape(3,-1)
    kernal=[k1,k2,k3,k4,k5,k6,k7,k8]
    for i in kernal:
        img_ki=signal.convolve2d(img,i,mode='same',boundary=boundfill,fillvalue=fill_value)
        list_edge.append(np.abs(img_ki))
    #第二步：对上述的8个方向上的边缘强度，对应位置取最大值作为最后的边缘强度
    edge=list_edge[0]
    for i in range(len(list_edge)):
        edge=edge*(edge>=list_edge[i])+list_edge[i]*(edge<list_edge[i])
        '''上面的算法用于求最大值，很高级'''
    edge[edge>255]=255#阈值分割
    edge=edge.astype(np.uint8)
    plt.imshow(edge,cmap='gray')
    plt.title('Robinson')
    plt.show()
    return edge
edge1=kirsch(img,boundfill='symm')
#简单的素描结果，对上面的结果反色处理
pencil1=255-edge1
plt.imshow(pencil, cmap='gray')
plt.title('pencil 1')
plt.show()
#canny算子
'''def nothing(*args):
    pass
thresh_low,thresh_high=1,1
max_thresh_low,max_thresh_high=255,255
cv2.namedWindow('canny',1)
cv2.createTrackbar('thresh_low','canny',thresh_low,max_thresh_low,nothing)
cv2.createTrackbar('thresh_high','canny',thresh_high,max_thresh_high,nothing)
while True:
    thresh_low=cv2.getTrackbarPos('thresh_low','canny')
    thresh_high=cv2.getTrackbarPos('thresh_high','canny')
    canny=np.zeros_like(img)
    canny=cv2.Canny(img,thresh_low,thresh_high,edges=canny,apertureSize=3,L2gradient=True)
    cv2.imshow('canny',canny)
    ch = cv2.waitKey(5)
    if ch == 27:  # 按下ESC键退出内循环
        break'''
#LOG(高斯拉普拉斯)边缘检测:
gauss=cv2.GaussianBlur(img,(3,3),0)
log=cv2.Laplacian(gauss,cv2.CV_32F)
log[log>255]=255
log[log<0]=0
log=log.astype(np.uint8)
plt.imshow(log, cmap='gray')
plt.title('LOG')
plt.show()

#DOG(高斯差分)边缘检测：
def DOG(img,size,sigma,k=1.6):
    img=np.copy(img).astype(np.float64)
    gauss1=cv2.GaussianBlur(img,size,sigma)
    gauss2=cv2.GaussianBlur(img,size,k*sigma)
    dog=gauss2-gauss1
    binary=np.copy(dog)
    binary[binary>0]=255
    binary[binary<=0]=0
    plt.imshow(dog,cmap='gray')
    plt.title('DOG')
    plt.show()
    plt.imshow(binary, cmap='gray')
    plt.title('DOG binary')
    plt.show()
    return dog,binary
DoG,binary_GoG=DOG(img,(7,7),2)
#Marr-Hildreth算法
#检测过零点
def zero_cross_mean(dog):
    zero_cross=np.zeros(dog.shape,np.uint8)
    ave=np.zeros(4)
    r,c=dog.shape
    for i in range(1,r-1):
        for j in range(1,c-1):
            ave[0]=np.mean(dog[i-1:i+1,j-1:j+1])
            ave[1] = np.mean(dog[i - 1:i + 1, j:j + 2])
            ave[2] = np.mean(dog[i:i + 2, j - 1: j + 1])
            ave[3] = np.mean(dog[i:i + 2, j: j + 2])
            if(np.min(ave)*np.max(ave)<0):
                zero_cross[i][j]=255
    return zero_cross
def marr_hildreth(img,size,sigma,k=1.6):#调节参数size和sigma都会影响结果
    dog,binary=DOG(img,size,sigma,k)
    zero_cross=zero_cross_mean(dog)
    plt.imshow(zero_cross,cmap='gray')
    plt.title('Marr Hildreth')
    plt.show()
    return zero_cross
marr=marr_hildreth(img,(37,37),2)


#梯度
grad_x=np.array([[-1,0],[0,1]],dtype=np.int)
grad_y=np.array([[0,-1],[1,0]],dtype=np.int)
grad_img_x=cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_16S,grad_x))#数据转换为uint8.对于大于255的自动截断
grad_img_y=cv2.convertScaleAbs(cv2.filter2D(img,cv2.CV_16S,grad_y))
res=np.hstack((img,grad_img_x))
cv2.imshow('grad img x',res)
res1=np.hstack((img,grad_img_y))
cv2.imshow('grad img y',res1)
roberts=cv2.addWeighted(grad_img_x,0.5,grad_img_y,0.5,0)
res2=np.hstack((img,roberts))
cv2.imshow('grad img',res2)

cv2.waitKey(0)
cv2.destroyAllWindows()
