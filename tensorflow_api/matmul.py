#!/usr/bin/python
#################################################################
#
#    file: matmul.py
#   usage: ./matmul.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-04 20:12:09
#
#################################################################
import tensorflow as tf

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 2])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[2, 3])
c = tf.matmul(a,b)
with tf.Session() as sess:
    print sess.run(c)
