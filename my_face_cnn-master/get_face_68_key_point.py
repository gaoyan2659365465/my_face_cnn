#打开视频，把识别到的人脸裁剪，打光，存储下来备用


import cv2
import os
import sys
import random
import pickle
import numpy as np
import dlib_68_key_point as db


def _get_image_fname(data_dir, image_format, index):
    fname = image_format % index
    return '%s/%s' % (data_dir, fname)



output_dir = './my_faces'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)



vc = cv2.VideoCapture('../FacialCapture-master/Test.mp4') #读入视频文件
c=1
 
if vc.isOpened(): #判断是否正常打开
    rval , img = vc.read()
else:
    rval = False
 
timeF = 1000  #视频帧计数间隔频率

while rval:   #循环读取视频帧
    rval, img = vc.read()
    img = np.rot90(img)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)#不进行缩放的话dlib检测不出来

    print('即将进入检测_', c)
    data = db.get_68_key_point(img)
    
    #获取图片路径
    pkl_fname = _get_image_fname(output_dir, 'my68_%08d.pkl', c)
    c = c + 1

    pkl_f = open(pkl_fname, 'wb')
    pickle.dump(data, pkl_f)
    pkl_f.close()

    cv2.waitKey(0)

vc.release()