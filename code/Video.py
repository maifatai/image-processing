'''
视频的读取
'''
import cv2
import numpy as np
cap=cv2.VideoCapture(r"Video.mp4")#创建摄像头对象
fps=cap.get(cv2.CAP_PROP_FPS)#视频中有多少帧
size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
code=cap.get(cv2.CAP_PROP_FOCUS)#视频的编码方式
numb=cap.get(cv2.CAP_PROP_FRAME_COUNT)#视频的总帧数
print(fps,size)
print(code,numb)
if cap.isOpened():
    open,frame=cap.read()
else:
    open=False
while open:
    ret,frame=cap.read()#读取视频，按帧读取
    if frame is None:
        break
    if ret==True:
        cv2.imshow('capture',frame)#显示每帧图像
        if cv2.waitKey(10)&0xff==ord('q'):
            break
cap.release()
cv2.destroyAllWindows()