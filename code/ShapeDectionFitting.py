import cv2
import numpy as np
import matplotlib.pyplot as plt
#最小外包旋转矩形
def rotate_rectangle():
    points=np.random.randint(100,300,(5,2),np.int32)
    rotate_rect=cv2.minAreaRect(points)#返回值为一个三元组，分别为旋转矩形的中心坐标，尺寸，旋转角度
    vertices=cv2.boxPoints(rotate_rect)#计算旋转矩形的四个顶点
    #根据四个顶点画出矩形
    img=np.zeros((400,400),np.uint8)
    for i in range(len(points)):
        cv2.circle(img,(points[i,0],points[i,1]),radius=2,color=255,thickness=2)
    for i in range(len(vertices)):
        p1=vertices[i,:]
        j=(i+1)%4
        p2=vertices[j,:]
        cv2.line(img,(int(p1[0]),int(p1[1])),(int(p2[0]),int(p2[1])),255,2)
    cv2.imshow('min rotate rect',img)
rotate_rectangle()
#最小外包圆
def min_circle():
    points1=np.random.randint(100,300,(5,2),np.int32)
    circle=cv2.minEnclosingCircle(points1)
    print(circle)
    img1=np.zeros((400,400),np.uint8)
    for i in range(len(points1)):
        cv2.circle(img1,(points1[i,0],points1[i,1]),radius=2,color=255,thickness=2)
    cv2.circle(img1,center=(int(circle[0][0]),int(circle[0][1])),radius=int(circle[1]),color=255,thickness=2)
    cv2.imshow('min closing circle',img1)
min_circle()
#最小外包直立矩形
def min_rect():
    points2=np.random.randint(100,300,(5,2),np.int32)
    rect=cv2.boundingRect(points2)
    print(rect)
    img2=np.zeros((400,400),np.uint8)
    for i in range(len(points2)):
        cv2.circle(img2,(points2[i,0],points2[i,1]),radius=2,color=255,thickness=2)
    cv2.rectangle(img2,(rect[0],rect[1]),(rect[2],rect[3]),color=255,thickness=2)
    cv2.imshow('min rect',img2)
min_rect()
#最小凸包
def min_convex_hull():
    img3=np.zeros((400,400),np.uint8)
    point3=np.random.randint(100,300,(80,2),np.int32)
    for i in range(80):
        cv2.circle(img3,(point3[i,0],point3[i,1]),radius=2,color=255,thickness=2)
    convex_hull=cv2.convexHull(point3)
    # defects=cv2.convexityDefects(point3,convex_hull)#轮廓的凸包缺陷
    # print(defects)
    #依次连接凸包的各个点
    k=convex_hull.shape[0]
    for i in range(k-1):
        cv2.line(img3,(convex_hull[i,0,0],convex_hull[i,0,1]),(convex_hull[i+1,0,0],convex_hull[i+1,0,1]),color=255,thickness=2)
    #还差首坐标和尾坐标相连
    cv2.line(img3,(convex_hull[k-1,0,0],convex_hull[k-1,0,1]),(convex_hull[0,0,0],convex_hull[0,0,1]),color=255,thickness=2)
    cv2.imshow('min convex hull',img3)
min_convex_hull()
#最小外包三角形
def min_tri():
    points4 = np.random.randint(100, 300, (5,1,2), np.int32)
    retval,triangle=cv2.minEnclosingTriangle(points4)
    print(retval,triangle)
    img4 = np.zeros((400, 400), np.uint8)
    for i in range(5):
        cv2.circle(img4, (points4[i,0, 0], points4[i,0,1]), radius=2, color=255, thickness=2)
    for i in range(3-1):
        cv2.line(img4,(int(triangle[i, 0, 0]),int(triangle[i, 0, 1])),
                 (int(triangle[i + 1, 0, 0]), int(triangle[i + 1, 0, 1])), color=255, thickness=2)
    # 还差首坐标和尾坐标相连
    cv2.line(img4, (int(triangle[3-1, 0, 0]), int(triangle[3-1, 0, 1])), (int(triangle[0, 0, 0]), int(triangle[0,0,1])),color=255,thickness=2)
    cv2.imshow('min triangle', img4)
min_tri()
#最优拟合椭圆
def fit_ellipse():
    img3 = np.zeros((400, 400), np.uint8)
    point3 = np.random.randint(100, 300, (10, 2), np.int32)
    for i in range(10):
        cv2.circle(img3, (point3[i, 0], point3[i, 1]), radius=2, color=255, thickness=2)
    elipse=cv2.fitEllipse(point3)#返回值为椭圆中心点的位置、长短轴长度，是为长短轴的直径，而非半径、中心旋转的角度
    cv2.ellipse(img3,(int(elipse[0][0]),int(elipse[0][1])),(int(elipse[1][0]/2),int(elipse[1][1]/2)),int(elipse[2]),0,360,color=255,thickness=2)
    cv2.imshow('fit ellipse', img3)
fit_ellipse()
def fit_line():
    img3 = np.zeros((400, 400), np.uint8)
    rows,cols=img3.shape[:2]
    point3 = np.random.randint(100, 300, (10, 2), np.int32)
    for i in range(10):
        cv2.circle(img3, (point3[i, 0], point3[i, 1]), radius=2, color=255, thickness=2)
    [vx,vy,x,y]=cv2.fitLine(point3,cv2.DIST_L2,0,0.01,0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(img3, (cols - 1, righty), (0, lefty), 255, 2)
    cv2.imshow('fit line',img3)
fit_line()
#霍夫直线检测
img=cv2.imread('lena.jpg',0)
edge=cv2.Canny(img,threshold1=60,threshold2=150,apertureSize=3)
lines=cv2.HoughLines(edge,1,np.pi/180,150)#标准霍夫直线检测
# lines1=cv2.HoughLinesP(edge,1,np.pi/180, 100)#
line=lines[:,0,:]
for r,theta in line[:]:
    # Stores the value of cos(theta) in a
    a = np.cos(theta)
    # Stores the value of sin(theta) in b
    b = np.sin(theta)
    # x0 stores the value rcos(theta)
    x0 = a * r
    # y0 stores the value rsin(theta)
    y0 = b * r
    # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    x1 = int(x0 + 1000 * (-b))
    # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    y1 = int(y0 + 1000 * (a))
    # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    x2 = int(x0 - 1000 * (-b))
    # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    y2 = int(y0 - 1000 * (a))
    # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
    # (0,0,255) denotes the colour of the line to be
    # drawn. In this case, it is red.
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
cv2.imshow('hough line',img)
#基于梯度的霍夫圆检测函数
src=cv2.imread('circle.jpg')
src_gray=cv2.cvtColor(src,cv2.COLOR_RGB2GRAY)
circle=cv2.HoughCircles(src_gray,cv2.HOUGH_GRADIENT,1,minDist=100,param1=150,param2=100,minRadius=100)
len=circle.shape[1]
for i in range(len):
    center=(int(circle[0,i,0]),int(circle[0,i,1]))
    radius=int(circle[0,i,2])
    cv2.circle(src,center,radius,(255,255,255),3)
cv2.imshow('hough circle',src)
#轮廓contours_img=[]
src=cv2.imread('shape.png',0)
src_gray=cv2.Canny(src,60,150,apertureSize=3)
cv2.imshow('canny shape',src_gray)
contours,h=cv2.findContours(src_gray,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#寻找轮廓，返回到contours的一个列表

for i in range(2):
    tmp=np.zeros_like(src_gray)
    cv2.drawContours(tmp,contours,i,255,2)
    circle=cv2.minEnclosingCircle(contours[i])
    cv2.circle(tmp,(int(circle[0][0]),int(circle[0][1])), radius=int(circle[1]), color=255, thickness=2)
    appro=cv2.approxPolyDP(contours[i],0.3,True)
    for j in range(appro.shape[0]-1):
        cv2.line(tmp,(appro[i,0,0],appro[i,0,1]),(appro[i+1,0,0],appro[i+1,0,1]),0,5)
    cv2.line(tmp,(appro[appro.shape[0]-1,0,0],appro[appro.shape[0]-1,0,1]),(appro[0,0,0],appro[0,0,1]),0,5)
    cv2.imshow('contour '+str(i),tmp)
#轮廓的周长
points=np.random.randint(100,300,(5,2),np.int32)
length1=cv2.arcLength(points,False)
length2=cv2.arcLength(points,True)
area=cv2.contourArea(points)
print(length1,length2,area)
#判断点集是否在轮廓内
dst=cv2.pointPolygonTest(points,(150,150),True)
print(dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

