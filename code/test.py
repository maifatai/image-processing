import cv2
import matplotlib.pyplot as plt
src=cv2.imread("text_image.tif",0)
plt.figure('histogram')
plt.title('plt：histogram')
plt.hist(src.ravel(),256)
plt.show()
hist=cv2.calcHist([src],[0],None,[256],[0,256])
plt.title('cv：histogram of lena')
plt.plot(hist)
plt.show()