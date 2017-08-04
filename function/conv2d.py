#!/usr/bin/python
#################################################################
#
#    file: conv2d.py
#   usage: ./conv2d.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-03 19:28:32
#
#################################################################
import tensorflow as tf
a = tf.Variable(tf.random_normal([1,27,27,3]))
b = tf.Variable(tf.random_normal([5,5,3,32]))
c = tf.nn.conv2d(a,b,strides=[1, 2, 2, 1],padding='VALID')
d = tf.nn.conv2d(a,b,strides=[1, 2, 2, 1],padding='SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print c.shape
    print d.shape
