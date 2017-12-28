#_*_ coding:utf-8 _*_
import tensorflow as tf
import collections
import numpy as np
from scipy.misc import imresize
from scipy.misc import imread
from tensorflow.python.framework.ops import convert_to_tensor
from caffe_classes import class_names

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32) #RGB

class BatchImage(
    collections.namedtuple("BatchImage",
                           ("initializer",
                            "img_batch",
                            "labels_batch"))):
    pass

class ImageDataGenerator(object):
    def __init__(self,hparams):
        self.txt_file = hparams.txt_file
        self.mode = hparams.mode
        self.batch_size = hparams.batch_size
        self.num_classes = hparams.num_classes
        self.buffer_size = 1000

        self._read_txt_file()
        self.img_paths = convert_to_tensor(self.img_paths, dtype = tf.string)
        self.labels = convert_to_tensor(self.labels, dtype = tf.int32)

    @classmethod
    def get_image(cls,filename):
        img = (imread(filename)[:,:,:3]).astype(np.float32)
        img = img - np.mean(img)
        img[:, :, 0], img[:, :, 2] = img[:, :, 2], img[:, :, 0]
        return img

    @classmethod
    def predict(cls,output):
        for i in range(output.shape[0]):
            print("image: %d" % i)
            index = np.argsort(output)[i,:] #返回排序后索引，从小到大拍讯
            for j in range(5):
                print("  ",class_names[index[-1-j]],output[i,index[-1-j]])

    def _read_txt_file(self):
        self.img_paths = list()
        self.labels = list()
        with open(self.txt_file,'r') as f:
            for line in f.readlines():
                items = line.strip('\n').split(' ')
                if len(items) > 1:
                    self.img_paths.append(items[0])
                    self.labels.append(int(items[1]))

    def _parse_function(self,filename, label):
        one_hot = tf.one_hot(label, self.num_classes)
        img = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img,channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr,one_hot

    def _parse_function_train(self, filename, label):
        return self._parse_function(filename,label)

    def _parse_function_inference(self, filename, label):
        return self._parse_function(filename,label)

    def get_iterator(self):
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        output_buffer_size = 100*self.batch_size
        data = data.shuffle(output_buffer_size)
        if self.mode == "training":
            data = data.map(self._parse_function_train,num_parallel_calls=8)
        elif self.mode == "inference":
            data = data.map(self._parse_function_inference,num_parallel_calls=8)
        else:
            raise ValueError("Invalid mode '%s'." % (self.mode))
        data = data.batch(self.batch_size)
        iterator = data.make_initializable_iterator()
        img_batch,labels_batch = iterator.get_next()
        return BatchImage(
            initializer = iterator.initializer,
            img_batch = img_batch,
            labels_batch = labels_batch)

'''
hparams = tf.contrib.training.HParams(
    txt_file = "imageNet",
    mode = "training",
    batch_size = 1,
    num_classes = 1000)
img = ImageDataGenerator(hparams)
iterator = img.get_iterator()
img_batch = iterator.img_batch
labels_batch = iterator.labels_batch
with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(iterator.initializer)
    for i in range(1):
        try:
            print(sess.run([img_batch,labels_batch]))
        except tf.errors.OutOfRangeError:
            print("OutOfRangeError")
            sess.run(iterator.initializer)'''
