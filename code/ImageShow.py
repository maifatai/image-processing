#-*- coding:utf-8 -*-
import cv2 as cv
if __name__=="__main__":
    img=cv.imread('lena.jpg',1)
    cv.imshow('lena',img)
    gray=cv.cvtColor(img,cv.COLOR_RGB2GRAY)#转化为灰度图
    cv.imshow('gray',gray)
    ret,dst=cv.threshold(gray,100,255,cv.THRESH_BINARY)#二值化
    cv.imshow('binary',dst)
    cv.waitKey(0)
    cv.destroyAllWindows()





