#根据之前生成的数据，保存成tensorflow专用的格式
#加载速度增加了
import argparse
import pickle
import tensorflow as tf
import os
import numpy as np




def _get_output_tfrecord_name(output_dir, basename, index):
    """
    根据编号获取一个tfrecord新名字
    """
    return '%s/%s_%08d.tfrecord' % (output_dir, basename, index) 

def loca_data(pkl_fname):
    """
    打开数据文件，返回数据
    """
    pkl_f = open(pkl_fname, 'rb')
    data = pickle.load(pkl_f)
    pkl_f.close()

    return data

def _get_image_fname(data_dir, image_format, index):
    """
    获取一个图片路径新名字
    """
    fname = image_format % index
    return '%s/%s' % (data_dir, fname)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))
def _floats_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help = "数据路径", default =
            "/home/gaoyan/文档/my_face_cnn-master/my_faces")
    parser.add_argument("--start", help = "开始的编号", default = 1)
    parser.add_argument("--end", help = "结束的编号", default = 3000)
    parser.add_argument(\
            "--image_format", \
            help = "image format can either be pil image file or pickle file",\
            default = 'my68_%08d.pkl')
    parser.add_argument("--tfrecord_dir", help = "tfreccord dir", default = "/home/gaoyan/文档/my_face_cnn-master/tfrecord")
    parser.add_argument("--tfrecord_fname", help = "tfrecord basename", default = "tf_record")#保存二进制文件的名称前缀
    parser.add_argument("--tfrecord_num", help = "store number of images per tfrecord", default = 1000)#每隔二进制文件要保存多少图片和模型


    args = parser.parse_args()


    idx = 0
    if not os.path.isdir(args.tfrecord_dir):
        os.makedirs(args.tfrecord_dir)
        print("创建路径 {}".format(args.tfrecord_dir))#如果没有定义保存路径就自动创建这个路径
    record_fname = _get_output_tfrecord_name(args.tfrecord_dir, args.tfrecord_fname, idx)#根据编号获取一个tfrecord新名字
    tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)#根据名字创建一个tfrecord对象用于之后的写入
    print("保存到 {}".format(record_fname))#提示当前保存到哪个包文件里面

    tmp = 0
    for ii in range(int(args.start), int(args.end) + 1):#从开始遍历到结束
        image_fname = _get_image_fname(args.data_dir, args.image_format, ii)#获取图片的路径
        
        if not os.path.isfile(image_fname) :
            print("数据文件 {} 没有找到".format(image_fname))#如果找不到这个数据就提示，然后退出本次循环
            continue

        if image_fname[-3:] == 'pkl':
            print(image_fname)
            data = loca_data(image_fname)#加载数据文件
            try:
                img = data[1]
            except:
                print("文件打开出错 {}".format(image_fname))#如果打不开这个图片就退出这次循环
                continue
            #img = Image.fromarray(np.uint8(img * 255))#给图片转换一下格式
        try:
            example = tf.train.Example(features = tf.train.Features(feature = {\
                        'image' : _bytes_feature(img.tobytes()), \
                        'v' : _floats_feature(data[0]) \
                        }))#把图片数据和模型数据结组，变成example格式
        except:
                print("文件打开出错 {}".format(image_fname))#如果打不开这个图片就退出这次循环
                continue
        tfrecord_writer.write(example.SerializeToString())#序列化为字符串，写入tfrecord

        tmp += 1
        if tmp % args.tfrecord_num == 0:#取模，每隔100就保存一次
            tfrecord_writer.close()#关闭保存文件
            idx += 1#序号加一，用于获取下一个名字
            record_fname = _get_output_tfrecord_name(args.tfrecord_dir, args.tfrecord_fname, idx)#获取下一个名字
            tfrecord_writer = tf.python_io.TFRecordWriter(record_fname)#写入到TFRecords文件
            print("保存到 {}".format(record_fname))
            print("已完成 {}".format(tmp / (float(int(args.end) - int(args.start) + 1))))#提示已完成的数量












if __name__  == "__main__":
    main()