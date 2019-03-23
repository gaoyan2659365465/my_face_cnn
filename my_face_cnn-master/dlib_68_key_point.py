# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_68_key_point(img):
    """
    根据图像利用dlib获取68个关键点，保存成列表输出
    x = [(索引0, 坐标0), (索引1, 坐标1)]
    y = [x, 灰度图]
    """
    data = []

    # 取灰度
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        #print("正在检查人脸")
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img,rects[i]).parts()])
        for idx, point in enumerate(landmarks):
            # 68点的坐标
            #pos = (point[0, 0], point[0, 1])
            data.append(point[0, 0])
            data.append(point[0, 1])
            bdata = np.reshape(data, [-1])
        data = [bdata, img_gray]
    return data

    #应该把没有识别出来的图片自动不保存