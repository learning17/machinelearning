#!/usr/bin/python
#################################################################
#
#    file: max_pool.py
#   usage: ./max_pool.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-03 20:05:07
#
#################################################################
import tensorflow as tf

a = tf.Variable(tf.random_normal([1,27,27,3]))
b = tf.nn.max_pool(a,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='VALID')
c = tf.nn.max_pool(a,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print b.shape
    print c.shape
