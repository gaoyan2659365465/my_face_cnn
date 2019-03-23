import cv2
import dlib_68_key_point as db
import numpy as np


#加载摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret, img = cap.read()
    if img is None:
        break
    #img = np.rot90(img)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)#不进行缩放的话dlib检测不出来

    data = db.get_68_key_point(img)
    
    try:
        bdata = np.reshape(data[0], [-1,2])
    except:
        print("1")
        continue
    bdata = bdata.tolist()
    idx = 0
    for pos in bdata:
        # 利用cv2.circle给每个特征点画一个圈，共68个 
        cv2.circle(img, (pos[0], pos[1]), 0, color=(0, 255, 0))
        # 利用cv2.putText输出1-68 
        #font = cv2.FONT_HERSHEY_SIMPLEX 
        #cv2.putText(img, str(idx+1), (pos[0], pos[1]), font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)
        idx = idx+1
    cv2.namedWindow("img", 2)
    cv2.imshow("img", img)
    cv2.waitKey(1)

cv2.release()