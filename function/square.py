#!/usr/bin/python
#################################################################
#
#    file: square.py
#   usage: ./square.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-21 16:14:12
#
#################################################################
import tensorflow as tf
import numpy as np

initial_x = [[1.,1.],[2.,2.]]
x = tf.Variable(initial_x,dtype=tf.float32)
x2 = tf.square(x)
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print sess.run(x2)

