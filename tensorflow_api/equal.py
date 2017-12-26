#!/usr/bin/python
#################################################################
#
#    file: equal.py
#   usage: ./equal.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-03 16:52:42
#
#################################################################
import tensorflow as tf

a = tf.constant([1,2,3,4,5,6],shape=[2,3])
b = tf.constant([1,3,2,4,5,7],shape=[2,3])
c = tf.equal(a,b)

with tf.Session() as sess:
    print sess.run(a)
    print ("##################################")
    print sess.run(b)
    print ("##################################")
    print sess.run(c)
