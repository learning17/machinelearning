#_*_ coding:utf-8 _*_
from synset import *
import collections
import tensorflow as tf
import os
import re
import random

class BatchImage(
    collections.namedtuple("BatchImage",
                           ("initializer",
                            "img_batch",
                            "labels_batch"))):
    pass

class ImageNetGenerator(object):
    def __init__(self, hparams):
        self.data_dir = hparams.data_dir
        self.txt_file = hparams.txt_file
        self.is_training = hparams.is_training
        self.num_classes = hparams.num_classes
        self.batch_size = hparams.batch_size
        self.shuffle = hparams.shuffle
        self.height = hparams.height
        self.width = hparams.width
        self._read_txt_file()
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype = tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype = tf.int32)
        self.thread_id = 0

    def _read_txt_file(self):
        self.img_paths = list()
        self.labels = list()
        file_path = os.path.join(self.data_dir, self.txt_file)
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                ext = os.path.splitext(line)[1]
                if ext != '.JPEG':
                    continue
                label_name = re.search(r'(n\d+)', line).group(1)
                label_index = synset_map[label_name]["index"]
                img_path = os.path.join(self.data_dir, line)
                self.labels.append(label_index)
                self.img_paths.append(img_path)


    def _distort_image(self, img):
        image = tf.image.random_flip_left_right(img)
        if self.thread_id % 2 == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        else:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        return tf.clip_by_value(image, 0.0, 1.0)

    def _image_preprocessing(self, img, label):
        one_hot = tf.one_hot(label, self.num_classes)
        resize_method = self.thread_id % 4
        image = tf.image.resize_images(img,[self.height, self.width],resize_method)
        image.set_shape([self.height, self.width, 3])
        if self.thread_id > 0:
            image = self._distort_image(image)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        self.thread_id += 1
        return image, one_hot

    def _read_img_file(self, filename, label):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, label

    def get_iterator(self):
        data_org = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))
        data_org = data_org.map(self._read_img_file, num_parallel_calls=8)
        data = data_org.map(self._image_preprocessing, num_parallel_calls=8)
        if self.is_training:
            for i in range(4):
                data_t = data_org.map(self._image_preprocessing, num_parallel_calls=8)
                data = data.concatenate(data_t)
        if self.shuffle:
            data = data.shuffle(100*self.batch_size)
        data = data.batch(self.batch_size)
        iterator = data.make_initializable_iterator()
        img_batch,labels_batch = iterator.get_next()
        return BatchImage(
            initializer = iterator.initializer,
            img_batch = img_batch,
            labels_batch = labels_batch)

    @classmethod
    def divide_train_dev_test(cls,
                              file_dir = "/data/ImageNet/ILSVRC2012_img_train.txt",
                              train_file = "/data/ImageNet/ILSVRC2012_img_train/train.txt",
                              dev_file = "/data/ImageNet/ILSVRC2012_img_train/dev.txt"):
        val_ratio = 0.05
        with open(file_dir,'r') as f, open(train_file,'w') as trainf, open(dev_file,'w') as devf:
            for line in f.readlines():
                line = line.strip()
                ran = random.random()
                if ran <= val_ratio:
                    devf.write(line)
                else:
                    trainf.write(line)

'''
hparams = tf.contrib.training.HParams(
    data_dir = "/data/ImageNet/ILSVRC2012_img_train",
    txt_file = "train.txt",
    is_training = True,
    num_classes = 1000,
    batch_size = 64,
    shuffle = True,
    height = 224,
    width = 224)
tr_data = ImageNetGenerator(hparams)
tr_iterator = tr_data.get_iterator()

step = 0
with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(tr_iterator.initializer)
    while True:
        print(step)
        step += 1
        img_batch,labels = sess.run([tr_iterator.img_batch, tr_iterator.labels_batch])
        print(img_batch.shape)
'''
