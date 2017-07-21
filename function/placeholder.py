#!/usr/bin/python
#################################################################
#
#    file: placeholder.py
#   usage: ./placeholder.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-21 15:07:33
#
#################################################################
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32,[None,10])
y = tf.matmul(x,x)
with tf.Session() as sess:
    rand_array = np.random.rand(10,10)
    print sess.run(y,feed_dict={x:rand_array})

