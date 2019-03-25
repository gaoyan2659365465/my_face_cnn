import tensorflow as tf
import pickle



def loca_data(pkl_fname):
    """
    打开数据文件，返回数据
    """
    pkl_f = open(pkl_fname, 'rb')
    data = pickle.load(pkl_f)
    pkl_f.close()

    return data


def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def _floats_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))


image_fname = "/home/gaoyan/文档/my_face_cnn/my_face_cnn-master/my_faces/" + "my68_00000001.pkl"


data = loca_data(image_fname)#加载数据文件
data = _floats_feature(data[0])
#print(type(data))
