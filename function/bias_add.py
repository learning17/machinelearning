#!/usr/bin/python
#################################################################
#
#    file: bias_add.py
#   usage: ./bias_add.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-21 15:33:42
#
#################################################################
import tensorflow as tf
import numpy as np

a = tf.constant([[1.0, 2.0],[1.0, 2.0],[1.0, 2.0]])
b = tf.constant([2.0,1.0])
c = tf.constant([1.0])
sess = tf.Session()
print sess.run(tf.nn.bias_add(a, b))
#print sess.run(tf.nn.bias_add(a,c)) error
print ("##################################")
print sess.run(tf.add(a, b))
print ("##################################")
print sess.run(tf.add(a, c))

