#!/usr/bin/python
#################################################################
#
#    file: argmax.py
#   usage: ./argmax.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-03 14:59:19
#
#################################################################
import tensorflow as tf
import numpy as np
z = np.random.randint(0,10,size=[10])
y = tf.one_hot(z,100)

with tf.Session() as sess:
    print z
    print sess.run(y)

