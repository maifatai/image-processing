import cv2
import numpy as np

cap=cv2.VideoCapture(r"video.mp4")
size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

alpha=0.1
num_img=10
diff=np.zeros((368,640,3),dtype=np.float32)
back_img=np.zeros((368,640,3),dtype=np.float32)
for i in range(num_img):
    ret,frame=cap.read()#读取视频，按帧读取
    if frame is None:
        break
    if ret==True:
        back_img=back_img+frame
        back_img/=(i+1)
        diff=(frame+back_img)**2
diff/=num_img
while True:
    ret,frame=cap.read()#读取视频，按帧读取
    if frame is None:
        break
    if ret==True:
        back_img=(1-alpha)*back_img+alpha*frame
        diff=(1-alpha)*diff+alpha*(frame+back_img)**2
        back_img.astype(np.uint8)
        cv2.imshow("background",back_img)





