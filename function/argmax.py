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
L = range(24)
a = tf.constant(L,shape=[2,3,4])
b = tf.argmax(a,axis=0)
c = tf.argmax(a,axis=1)
d = tf.argmax(a,axis=2)
with tf.Session() as sess:
    print sess.run(a)
    print ("##################################")
    print sess.run(b)
    print ("##################################")
    print sess.run(c)
    print ("##################################")
    print sess.run(d)

