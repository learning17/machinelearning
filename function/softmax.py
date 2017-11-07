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

b = tf.constant([0, 1, 1, 1],dtype=tf.float32,shape=[2,2 ])
p = tf.nn.softmax(b)
with tf.Session() as sess:
    print sess.run(b)
    print sess.run(p)
