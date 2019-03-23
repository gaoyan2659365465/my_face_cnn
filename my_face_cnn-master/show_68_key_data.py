#用来测试和显示已经序列化的图片数据
#打开一个文件，根据已经设定好的格式加载里面的图片

import cv2
import pickle
import numpy as np


pkl_fname = 'my_faces/my68_00001000.pkl'
output_dir = './my_faces'
def _get_image_fname(data_dir, image_format, index):
    fname = image_format % index
    return '%s/%s' % (data_dir, fname)

"""
data数据格式
x = [(索引0, 坐标0), (索引1, 坐标1)]
y = [x, 灰度图]
"""

class Show_data():
    def __init__(self):
        index = 0
        while True:
            index = index+1
            pkl_fname = _get_image_fname(output_dir, 'my68_%08d.pkl', index)
            print(pkl_fname)
            data = self.loacl_data(pkl_fname)
            try:
                img = self.circle_img(data[1],data[0])
            except:
                continue
            self.show_img(img)
            
    
    def loacl_data(self, pkl_fname):
        """
        加载图片文件数据
        """
        pkl_f = open(pkl_fname, 'rb')
        data = pickle.load(pkl_f)
        pkl_f.close()

        return data
    
    def show_img(self, img):
        cv2.namedWindow("img", 2)
        cv2.imshow("img", img)
        cv2.waitKey(1)
    
    def circle_img(self, img, key_data):
        bdata = np.reshape(key_data, [-1,2])
        bdata = bdata.tolist()
        idx = 0
        for pos in bdata:
            # 利用cv2.circle给每个特征点画一个圈，共68个 
            cv2.circle(img, (pos[0], pos[1]), 2, color=(0, 255, 0))
            # 利用cv2.putText输出1-68 
            font = cv2.FONT_HERSHEY_SIMPLEX 
            cv2.putText(img, str(idx+1), (pos[0], pos[1]), font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)
            idx = idx+1
        return img

Show_data()