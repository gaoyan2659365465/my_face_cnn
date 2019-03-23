from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from ModelConfig import ModelConfig
from VisualModule import Module

tf.logging.set_verbosity(tf.logging.INFO)
tf.reset_default_graph()


# 打开摄像头 参数为输入流，可以为摄像头或视频文件
camera = cv2.VideoCapture('/home/gaoyan/文档/my_face_cnn-master/Test.mp4')
#camera = cv2.VideoCapture(0)



FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string(\
        "checkpoint_dir", \
        "/home/gaoyan/文档/my_face_cnn-master/train_dir", \
        "path to checkpoint file")

with tf.Session() as sess:
    model_config = ModelConfig()
    model = Module(model_config, mode="inference")
    model.build()

    saver = tf.train.Saver()#加载保存模块
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=FLAGS.checkpoint_dir)#寻找存档路径
    saver.restore(sess, ckpt.model_checkpoint_path)#载入存档


    



    while True:
        # 从摄像头读取照片
        success, img = camera.read()
        """
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray_img = np.rot90(gray_img)
        gray_img=Image.fromarray(gray_img)

        bbox = (50, 136, 662, 1042)
        face = gray_img.crop(bbox).resize((256, 256), Image.BILINEAR)#修减
        """

        img = np.rot90(img)
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)#不进行缩放的话dlib检测不出来
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        moxing = np.squeeze(sess.run(model.prediction, \
                    feed_dict = { model.image_feed : img_gray.tobytes() } ))#将图片传入占位符
        
        moxing = moxing.reshape((68, 2))


        gray_img = np.array(img_gray)#image转numpy
        moxing = np.round(moxing).astype(np.int32)
        for i in range(0, moxing.shape[0], 2):
            st = moxing[i, :2]
            gray_img = cv2.circle(gray_img,(st[0], st[1]), 1, (255,0,0), -1)

        cv2.imshow('image', gray_img)
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break