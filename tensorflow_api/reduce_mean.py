#!/usr/bin/python
#################################################################
#
#    file: reduce_mean.py
#   usage: ./reduce_mean.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-21 15:44:05
#
#################################################################
import tensorflow as tf
import numpy as np

initial = [[1.,1.],[2.,2.]]
x = tf.Variable(initial,dtype=tf.float32)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(tf.reduce_mean(x)))
    print(sess.run(tf.reduce_mean(x,0,keep_dims=True))) #Column
    print(sess.run(tf.reduce_mean(x,1,keep_dims=True))) #row



