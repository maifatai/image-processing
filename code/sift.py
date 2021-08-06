import numpy as np
import cv2
import matplotlib.pyplot as plt
def brute_force_maching(img1,img2):
    '''
    BFmatcher（Brute-Force Matching）暴力匹配，
    应用BFMatcher.knnMatch( )函数来进行核心的匹配，
    knnMatch（k-nearest neighbor classification）k近邻分类算法。
    经检验 BFmatcher在做匹配时会耗费大量的时间。
    '''
    # gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift=cv2.SIFT_create()
    kp1,dst1=sift.detectAndCompute(img1,None)
    kp2,dst2=sift.detectAndCompute(img2,None)#dst为描述子
    merge=np.hstack((img1,img2))
    img3=cv2.drawKeypoints(img1,kp1,img1,color=(255,255,255))
    img4=cv2.drawKeypoints(img2,kp2,img2,color=(255,255,255))#画出特征点
    key_merge=np.hstack((img3,img4))
    # cv2.imshow("point",key_merge)
    #opencv 的接口使用BGR模式，matplotlib.pyplot 接口使用的是RGB模式。
    b, g, r = cv2.split(key_merge)
    key_merge= cv2.merge([r, g, b])
    plt.imshow(key_merge)
    plt.title("BF point")
    plt.show()
    bf=cv2.BFMatcher()#BFmatch匹配
    match=bf.knnMatch(dst1,dst2,k=2)#k表示KNN中的
    good=[]
    #调整比率
    for m,n in match:
        if m.distance<0.7*n.distance:
            good.append([m])
    img5=cv2.drawMatchesKnn(img1,kp1,img2,kp2,match,None,flags=2)
    b, g, r = cv2.split(img5)
    img5= cv2.merge([r, g, b])
    plt.imshow(img5)
    plt.title("BFmatch")
    plt.show()
    img6=cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    b, g, r = cv2.split(img6)
    img6= cv2.merge([r, g, b])
    plt.imshow(img6)
    plt.title("BFmatch good")
    plt.show()
'''
FLANN(Fast_Library_for_Approximate_Nearest_Neighbors)快速最近邻搜索包，它是一个对大数据集和高维特征进行最近邻搜索的算法的集合,而且这些算法都已经被优化过了。在面对大数据集时它的效果要好于 BFMatcher。
经验证，FLANN比其他的最近邻搜索软件快10倍。使用 FLANN 匹配,我们需要传入两个字典作为参数。这两个用来确定要使用的算法和其他相关参数等。
第一个是 IndexParams。
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 。
这里使用的是KTreeIndex配置索引，指定待处理核密度树的数量（理想的数量在1-16）。
第二个字典是SearchParams。
search_params = dict(checks=100)用它来指定递归遍历的次数。值越高结果越准确，但是消耗的时间也越多。实际上，匹配效果很大程度上取决于输入。
5kd-trees和50checks总能取得合理精度，而且短时间完成。在之下的代码中，丢弃任何距离大于0.7的值，则可以避免几乎90%的错误匹配，但是好的匹配结果也会很少。
'''
def FLANN(img1,img2):
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # FLANN 参数设计
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    kp1, dst1 = sift.detectAndCompute(img1, None)
    kp2, dst2 = sift.detectAndCompute(img2, None)  # dst为描述子
    merge = np.hstack((img1, img2))
    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 255, 255))
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 255, 255))  # 画出特征点
    key_merge = np.hstack((img3, img4))
    # cv2.imshow("point",key_merge)
    # opencv 的接口使用BGR模式，matplotlib.pyplot 接口使用的是RGB模式，所以相同的三维数组，显示不同。
    b, g, r = cv2.split(key_merge)
    key_merge = cv2.merge([r, g, b])
    plt.imshow(key_merge)
    plt.title("FLANN point")
    plt.show()
    match = flann.knnMatch(dst1, dst2, k=2)
    matchesMask = [[0, 0] for i in range(len(match))]
    good = []
    # 调整比率
    '''
    在lowe论文中，Lowe推荐ratio的阈值为0.8，但作者对大量任意存在尺度、旋转和亮度变化的两幅图片进行匹配，
    结果表明ratio取值在0. 4~0. 6 之间最佳，小于0. 4的很少有匹配点，大于0. 6的则存在大量错误匹配点，
    所以建议ratio的取值原则如下:
    ratio=0. 4：对于准确度要求高的匹配；
    ratio=0. 6：对于匹配点数目要求比较多的匹配；
    ratio=0. 5：一般情况下。
    '''
    for m, n in match:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, match, None, flags=2)
    b, g, r = cv2.split(img5)
    img5 = cv2.merge([r, g, b])
    plt.imshow(img5)
    plt.title("FLANN")
    plt.show()
    img6 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    b, g, r = cv2.split(img6)
    img6 = cv2.merge([r, g, b])
    plt.imshow(img6)
    plt.title("FLANN good")
    plt.show()
'''
SURF全称为“加速稳健特征”（Speeded Up Robust Feature）,
不仅是尺度不变特征，而且是具有较高计算效率的特征。
可被认为SURF是尺度不变特征变换算法（SIFT算法）的加速版。
SURF最大的特征在于采用了haar特征以及积分图像的概念，SIFT采用的是DoG图像，而SURF采用的是Hessian矩阵（SURF算法核心）行列式近似值图像。
SURF借鉴了SIFT算法中简化近似的思想，实验证明，SURF算法较SIFT算法在运算速度上要快3倍，综合性优于SIFT算法。
'''
def SURF(img1,img2):
    # gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    # FLANN 参数设计
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    kp1, dst1 = sift.detectAndCompute(img1, None)
    kp2, dst2 = sift.detectAndCompute(img2, None)  # dst为描述子
    merge = np.hstack((img1, img2))
    img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 255, 255))
    img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 255, 255))  # 画出特征点
    key_merge = np.hstack((img3, img4))
    # cv2.imshow("point",key_merge)
    # opencv 的接口使用BGR模式，matplotlib.pyplot 接口使用的是RGB模式
    b, g, r = cv2.split(key_merge)
    key_merge = cv2.merge([r, g, b])
    plt.imshow(key_merge)
    plt.title("point")
    plt.show()
    match = flann.knnMatch(dst1, dst2, k=2)
    matchesMask = [[0, 0] for i in range(len(match))]
    good = []
    # 调整比率
    '''
    在lowe论文中，Lowe推荐ratio的阈值为0.8，但作者对大量任意存在尺度、旋转和亮度变化的两幅图片进行匹配，
    结果表明ratio取值在0. 4~0. 6 之间最佳，小于0. 4的很少有匹配点，大于0. 6的则存在大量错误匹配点，
    所以建议ratio的取值原则如下:
    ratio=0. 4：对于准确度要求高的匹配；
    ratio=0. 6：对于匹配点数目要求比较多的匹配；
    ratio=0. 5：一般情况下。
    '''
    for m, n in match:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, match, None, flags=2)
    b, g, r = cv2.split(img5)
    img5 = cv2.merge([r, g, b])
    plt.imshow(img5)
    plt.title("SURF")
    plt.show()
    img6 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    b, g, r = cv2.split(img6)
    img6 = cv2.merge([r, g, b])
    plt.imshow(img6)
    plt.title("SURF good")
    plt.show()
'''
ORB(Oriented Fast and Rotated BRIEF)，结合Fast与Brief算法，并给Fast特征点增加了方向性，
使得特征点具有旋转不变性，并提出了构造金字塔方法，解决尺度不变性，但文章中没有具体详述。
特征提取是由FAST（Features from Accelerated Segment Test）算法发展来的，特征点描述是根据BRIEF（Binary Robust Independent Elementary Features）特征描述算法改进的。
ORB特征是将FAST特征点的检测方法与BRIEF特征描述子结合起来，并在它们原来的基础上做了改进与优化。
ORB主要解决BRIEF描述子不具备旋转不变性的问题。实验证明，ORB远优于之前的SIFT与SURF算法，ORB算法的速度是sift的100倍，是surf的10倍。
'''
def ORB(img1,img2):
    '''
    BFmatcher（Brute-Force Matching）暴力匹配，
    应用BFMatcher.knnMatch( )函数来进行核心的匹配，
    knnMatch（k-nearest neighbor classification）k近邻分类算法。
    经检验 BFmatcher在做匹配时会耗费大量的时间。
    '''
    # gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    # gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sift=cv2.ORB_create()
    kp1,dst1=sift.detectAndCompute(img1,None)
    kp2,dst2=sift.detectAndCompute(img2,None)#dst为描述子
    merge=np.hstack((img1,img2))
    img3=cv2.drawKeypoints(img1,kp1,img1,color=(255,255,255))
    img4=cv2.drawKeypoints(img2,kp2,img2,color=(255,255,255))#画出特征点
    key_merge=np.hstack((img3,img4))
    # cv2.imshow("point",key_merge)
    #opencv 的接口使用BGR模式，matplotlib.pyplot 接口使用的是RGB模式。
    b, g, r = cv2.split(key_merge)
    key_merge= cv2.merge([r, g, b])
    plt.imshow(key_merge)
    plt.title("OBR  point")
    plt.show()
    bf=cv2.BFMatcher()#BFmatch匹配
    match=bf.knnMatch(dst1,dst2,k=2)#k表示KNN中的
    good=[]
    #调整比率
    for m,n in match:
        if m.distance<0.7*n.distance:
            good.append([m])
    img5=cv2.drawMatchesKnn(img1,kp1,img2,kp2,match,None,flags=2)
    b, g, r = cv2.split(img5)
    img5= cv2.merge([r, g, b])
    plt.imshow(img5)
    plt.title("OBR")
    plt.show()
    img6=cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
    b, g, r = cv2.split(img6)
    img6= cv2.merge([r, g, b])
    plt.imshow(img6)
    plt.title("ORB good")
    plt.show()
src1=cv2.imread('lena.jpg')
src2=cv2.imread('rotate.jpg')
brute_force_maching(src1,src2)
FLANN(src1,src2)
# SURF(src1,src2)

#FAST特征描述
fast=cv2.FastFeatureDetector_create()#获取FAST角点探测器
kp=fast.detect(src1,None)#描述符
img = cv2.drawKeypoints(src1,kp,src1,color=(255,255,0))#画到img上面
print ("Threshold: ", fast.getThreshold())#输出阈值
print ("nonmaxSuppression: ", fast.getNonmaxSuppression())#是否使用非极大值抑制
print ("Total Keypoints with nonmaxSuppression: ", len(kp))#特征点个数
cv2.imshow('fast',img)
cv2.waitKey(0)
cv2.destroyAllWindows()




