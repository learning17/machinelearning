#!/usr/bin/python
#################################################################
#
#    file: relu.py
#   usage: ./relu.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-04 19:56:45
#
#################################################################
import tensorflow as tf

a = tf.constant([1,-2,0,4,-5,6])
b = tf.nn.relu(a)
with tf.Session() as sess:
    print sess.run(b)
