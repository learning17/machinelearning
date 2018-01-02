#_*_ coding:utf-8 _*_
from alexnet import AlexNet
from generate_alexnet import ImageDataGenerator
from caffe_classes import class_names
import tensorflow as tf
import numpy as np
import sys

if __name__ == '__main__':
    test_x = ImageDataGenerator.get_image(sys.argv[1])
    X = tf.placeholder(tf.float32, [None, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)
    model = AlexNet(X,keep_prob)
    y_ = model.create_model()
    prob = tf.nn.softmax(y_)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.initial_weights(sess)
        y_,output =  sess.run([y_,prob],feed_dict={X: [test_x], keep_prob: 1})
        ImageDataGenerator.predict(output)

