## dirty config file
class ModelConfig(object):
    def __init__(self):
        
        self.image_feature_name = 'image'

        self.predict_feature_name = 'v'

        #self.num_weight = 51
        self.predict_num = 136#需要修改
        self.key_num = 68*2

        self.image_height = 640
        self.image_width = 360

        self.file_pattern = "/home/gaoyan/文档/my_face_cnn-master/tfrecord/tf_record*"

        self.train_dir = "./train_dir"

        #保持最大的检查点
        self.max_checkpoint_keep = 10

        #优化器
        self.optimizer = 'Adam'

        #一个从队列中出队新的批次的大小
        self.batch_size = 16

        #剪裁渐变
        self.clip_gradients = 5.
        
