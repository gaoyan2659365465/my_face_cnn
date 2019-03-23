import tensorflow as tf

#1定义文件名队列ft.train.string_input_producer
#2定义tf.TFRecordReader
#3读取序列化的example
#4调用tf.tf.parse_single_example解析example，得到features
#5从features获取具体的数据，如果是图像，进行解码和reshape（还可以进行相关的预处理）


class Module(object):
    def __init__(self, config, mode):
        assert mode in {'train', 'inference'}
        self.mode = mode
        self.config = config
        self.reader = tf.TFRecordReader()
    
    def build_inputs(self):
        if self.mode == 'train':
            data_files = []
            for pattern in self.config.file_pattern.split(','):
                print(tf.gfile.Glob(pattern))
                data_files.extend(tf.gfile.Glob(pattern))#返回带着所有pattern文件的列表

            print("训练的记录文件编号 : {}".format(len(data_files)))

            filename_queue = tf.train.string_input_producer( \
                    data_files, \
                    shuffle = True, \
                    capacity = 16)#输出字符串到一个输入管道队列
                                  #shuffle布尔值。如果为true，则在每个epoch内随机打乱顺序。
                                  #capacity一个整数，设置队列容量。
            _, example = self.reader.read(filename_queue)

            proto_value = tf.parse_single_example(\
                    example, \
                    features = { \
                    self.config.image_feature_name: tf.FixedLenFeature([], dtype=tf.string),\
                    self.config.predict_feature_name: tf.FixedLenFeature([self.config.predict_num], dtype=tf.float32) \
                   })#将数据键值映射成tensor值image和v
                   #tf.FixedLenFeature 返回的是一个定长的tensor
                   #tf.VarLenFeature 返回的是一个不定长的sparse tensor，用于处理可变长度的输入，在处理c t c 问题时，会用到tf.VallenFeature解析存储在tfrecord中的label。

            image = proto_value[self.config.image_feature_name]
            weight = proto_value[self.config.predict_feature_name]
            image = self.process_image(image)
            #weight = self.process_weight(weight)

            self.image, self.weight = tf.train.batch_join([(image, weight)], batch_size = self.config.batch_size)
            #运行一个tensor列表填充队列以创建example batches
            print('图片大小{}'.format(self.image.get_shape()))

        else:
            self.image_feed = tf.placeholder(dtype=tf.string, shape=[], name='image_feed')#占位符，在sess.run(output, feed_dict
            self.image = tf.expand_dims(self.process_image(self.image_feed),0)

    def process_image(self, im_str):
        """
        给图片数据转换格式，更改大小
        """
        image = tf.reshape(tf.decode_raw(im_str, out_type=tf.uint8), (self.config.image_width, self.config.image_height, 1))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images( \
                image, \
                size=[self.config.image_height, self.config.image_width],\
                method=tf.image.ResizeMethod.BILINEAR)
        return image

    def process_weight(self, im_str):
        """
        给68点数据转换格式
        """
        image = tf.reshape(tf.decode_raw(im_str, out_type=tf.uint8), (self.config.image_width, self.config.image_height, 1))
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images( \
                image, \
                size=[self.config.image_height, self.config.image_width],\
                method=tf.image.ResizeMethod.BILINEAR)
        return image

    def setup_global_step(self):
        """
        创建一个学习率变量，以备训练
        """
        self.global_step = tf.Variable( \
                initial_value=0, \
                name = 'global_step', \
                trainable = False, \
                collections = [tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    def build_model(self):
        with tf.variable_scope('face'):
            conv1a = tf.contrib.layers.conv2d( \
                    self.image, \
                    64, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False), \
                    scope = "conv1a", \
                    )
            conv1b = tf.contrib.layers.conv2d( \
                    conv1a, \
                    64, \
                    kernel_size = [3, 3], \
                    stride = 1 , \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False), \
                    scope = "conv1b",\
                    )
            conv2a = tf.contrib.layers.conv2d( \
                    conv1b, \
                    96, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False),\
                    scope = "conv2a" ,\
                    )
            conv2b = tf.contrib.layers.conv2d( \
                    conv2a, \
                    96, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False),\
                    scope = "conv2b")
            conv3a = tf.contrib.layers.conv2d( \
                    conv2b, \
                    144, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False), \
                    scope = 'conv3a')
            conv3b = tf.contrib.layers.conv2d( \
                    conv3a, \
                    144, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                    weights_initializer = tf.contrib.layers.xavier_initializer(uniform = False), \
                    scope = 'conv3b')
            print("conv3b 的形状结构是 {}".format(conv3b.get_shape()))
            conv4a = tf.contrib.layers.conv2d( \
                    conv3b, \
                    216, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    scope = 'conv4a')
            conv4b = tf.contrib.layers.conv2d( \
                    conv4a, \
                    216, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                    scope = 'conv4b')
            conv5a = tf.contrib.layers.conv2d( \
                    conv4b, \
                    324, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    scope = 'conv5a')
            conv5b = tf.contrib.layers.conv2d( \
                    conv5a, \
                    324, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                    scope = 'conv5b')
            conv6a = tf.contrib.layers.conv2d( \
                    conv5b, \
                    486, \
                    kernel_size = [3, 3], \
                    stride = 2, \
                    scope = 'conv6a')

            conv6b = tf.contrib.layers.conv2d( \
                    conv6a, \
                    486, \
                    kernel_size = [3, 3], \
                    stride = 1, \
                    scope = 'conv6b')
            print("conv6b 的形状结构是 {}".format(conv6b.get_shape()))
            drop = tf.contrib.layers.dropout(\
                    conv6b, \
                    0.8, \
                    is_training = (self.mode == 'train'), \
                    scope = 'dropout')#如果是训练模式就dropout，否则不工作（防止神经网络过拟合）
            fc = tf.contrib.layers.fully_connected( \
                    inputs = drop, \
                    num_outputs = self.config.key_num, \
                    activation_fn = None, \
                    scope = 'fully_connected')
            print("fc 形状结构是 {}".format(fc.get_shape()))
            output = tf.contrib.layers.flatten(fc)
            output = tf.contrib.layers.fully_connected( \
                    inputs = output, \
                    num_outputs = self.config.key_num, \
                    activation_fn = None, \
                    scope = 'output')

            if self.mode == 'inference' :
                self.prediction = output
            else :
                losses = tf.square(output - self.weight)
                batch_loss = tf.reduce_sum(losses)
                tf.losses.add_loss(batch_loss)
                tf.summary.scalar('losses/batch_loss', batch_loss)
                self.total_loss = tf.losses.get_total_loss()
                tf.summary.scalar('losses/total_loss', self.total_loss)
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.op.name, var)


    def build(self):
        if self.mode == 'train' :
            self.setup_global_step()
        self.build_inputs()
        self.build_model()


"""
tf.contrib.layers.conv2d
常用的参数说明如下：

    inputs:形状为[batch_size, height, width, channels]的输入。
    num_outputs：代表输出几个channel。这里不需要再指定输入的channel了，因为函数会自动根据inpus的shpe去判断。
    kernel_size：卷积核大小，不需要带上batch和channel，只需要输入尺寸即可。[5,5]就代表5x5的卷积核，如果长和宽都一样，也可以只写一个数5.
    stride：步长，默认是长宽都相等的步长。卷积时，一般都用1，所以默认值也是1.如果长和宽都不相等，也可以用一个数组[1,2]。
    padding：填充方式，'SAME'或者'VALID'。
    activation_fn：激活函数。默认是ReLU。也可以设置为None
    weights_initializer：权重的初始化，默认为initializers.xavier_initializer()函数。
    weights_regularizer：权重正则化项，可以加入正则函数。biases_initializer：偏置的初始化，默认为init_ops.zeros_initializer()函数。
    biases_regularizer：偏置正则化项，可以加入正则函数。
    trainable：是否可训练，如作为训练节点，必须设置为True，默认即可。如果我们是微调网络，有时候需要冻结某一层的参数，则设置为False。

"""