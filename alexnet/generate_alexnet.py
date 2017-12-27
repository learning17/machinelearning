#_*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np
import collections
from tensorflow.python.framework.ops import convert_to_tensor

IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

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

    def _read_txt_file(self):
        self.img_paths = list()
        self.labels = list()
        with open(self.txt_file,'r') as f:
            for line in f.readlines():
                items = line.strip('\n').split(' ')
                if len(items) > 1:
                    self.img_paths.append(items[0])
                    self.labels.append(int(items[1]))

    def get_iterator(self):
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        data = data.batch(self.batch_size)
        iterator = data.make_initializable_iterator()
        img_batch,labels_batch = iterator.get_next()
        return BatchImage(
            initializer = iterator.initializer,
            img_batch = img_batch,
            labels_batch = labels_batch)

hparams = tf.contrib.training.HParams(
    txt_file = "imageNet",
    mode = "train",
    batch_size = 2,
    num_classes = 1000)
img = ImageDataGenerator(hparams)
iterator = img.get_iterator()
img_batch = iterator.img_batch
labels_batch = iterator.labels_batch
with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(iterator.initializer)
    print(sess.run(img_batch))
    print(sess.run(labels_batch))
