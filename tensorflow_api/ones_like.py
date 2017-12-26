#!/usr/bin/python
#################################################################
#
#    file: ones_like.py
#   usage: ./ones_like.py
#   brief:
#  author: ***
#   email: ***
# created: 2017-11-24 10:37:59
#
#################################################################
import tensorflow as tf

tensor = tf.constant([[1,1,0],[3,0,4]])
one = tf.ones_like(tensor)
with tf.Session() as sess:
    print(sess.run(one))
