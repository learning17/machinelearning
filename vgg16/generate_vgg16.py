#_*_ coding:utf-8 _*_
import tensorflow as tf
import collections
import numpy as np
from scipy.misc import imresize
from scipy.misc import imread
from tensorflow.python.framework.ops import convert_to_tensor
from imagenet_classes import class_names
import os
import random

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
        self.shuffle = hparams.shuffle
        self.buffer_size = 1000

        self._read_txt_file()
        self.img_paths = convert_to_tensor(self.img_paths, dtype = tf.string)
        self.labels = convert_to_tensor(self.labels, dtype = tf.int32)

    @classmethod
    def get_image(cls,filename):
        img = (imread(filename)[:,:,:3]).astype(np.float32)
        img = imresize(img,[224,224])
        #img = img/127.5 - 1.
        img = img - np.mean(img)
        img = img[:, :, ::-1]
        return img

    @classmethod
    def predict(cls,output):
        for i in range(output.shape[0]):
            print("image: %d" % i)
            index = np.argsort(output)[i,:] #返回排序后索引，从小到大拍讯
            for j in range(5):
                print("  ",class_names[index[-1-j]],output[i,index[-1-j]])

    @classmethod
    def get_image_path_label(cls, file_dir='data/train/', train_file='train.txt',
                             val_file = 'val.txt', pref_path='data/train/'):
        label_dir = {'cat':0, 'dog':1}
        img_format = ['jpg', 'png']
        val_ratio = 0.15
        with open(train_file,'w') as trainf,open(val_file,'w') as valf:
            for filename in os.listdir(file_dir):
                segs = filename.split('.')
                if len(segs) == 3 and segs[0] in label_dir and segs[2] in img_format:
                    label = label_dir.get(segs[0],-1)
                else:
                    label = -1
                    print("illegal image filename:%s" % filename)
                    continue
                ran = random.random()
                line = "{}{} {}\n".format(pref_path,filename,label)
                if ran <= val_ratio:
                    valf.write(line)
                else:
                    trainf.write(line)
            valf.close()
            trainf.close()

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
        img_decoded = tf.image.decode_jpeg(img,channels=3)
        img_resized = tf.image.resize_images(img_decoded, [224, 224])
        #img_centered = tf.subtract(tf.divide(img_resized,127.5),1)
        img_centered = tf.subtract(img_resized, tf.reduce_mean(img_resized))
        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]
        return img_bgr,one_hot

    def _parse_function_train(self, filename, label):
        return self._parse_function(filename,label)

    def _parse_function_inference(self, filename, label):
        return self._parse_function(filename,label)

    def get_iterator(self):
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        if self.shuffle:
            data = data.shuffle(100*self.batch_size)
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
    txt_file = "train.txt",
    mode = "training",
    batch_size = 2,
    num_classes = 2,
    shuffle = True)
tr_data = ImageDataGenerator(hparams)
tr_iterator = tr_data.get_iterator()

with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(tr_iterator.initializer)
    img_batch = sess.run(tr_iterator.img_batch)
    print(img_batch)
'''
