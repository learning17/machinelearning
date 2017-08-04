#!/usr/bin/python
#################################################################
#
#    file: dropout.py
#   usage: ./dropout.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-08-03 21:24:03
#
#################################################################
import tensorflow as tf

a = tf.constant([1,2,3,4,5,6],shape=[2,3],dtype=tf.float32)
b = tf.placeholder(tf.float32)
c = tf.nn.dropout(a,b,[2,1],1)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print sess.run(c,feed_dict={b:0.75})

