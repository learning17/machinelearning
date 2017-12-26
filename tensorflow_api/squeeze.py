#_*_ coding:utf-8 _*_
#################################################################
#
#    file: squeeze.py
#   usage: ./squeeze.py
#   brief:
#  author: ***
#   email: ***
# created: 2017-12-26 17:05:22
#
#################################################################
import tensorflow as tf
'''
Removes dimensions of size 1 from the shape of a tensor
'''
a = tf.constant(1,shape=[1, 2, 1, 3, 1, 1])
b = tf.squeeze(a)
c = tf.squeeze(a, [2, 4])

with tf.Session() as sess:
    print(a.shape)
    print(b.shape)
    print(c.shape)

