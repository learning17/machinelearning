#!/usr/bin/python
#################################################################
#
#    file: cast.py
#   usage: ./cast.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-03 17:03:15
#
#################################################################
import tensorflow as tf

a = tf.constant([True,False,True,False,False,True],shape=[2,3])
b = tf.cast(a,dtype=tf.float32)

with tf.Session() as sess:
    print sess.run(a)
    print ("##################################")
    print sess.run(b)
