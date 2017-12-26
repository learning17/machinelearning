#!/usr/bin/python
#################################################################
#
#    file: sigmoid_cross_entropy_with_logits.py
#   usage: ./sigmoid_cross_entropy_with_logits.py
#   brief:
#  author: ***
#   email: ***
# created: 2017-11-24 11:10:25
#
#################################################################
import tensorflow as tf
x = tf.constant([1,2,3,4,5,6,7],dtype=tf.float64)
y = tf.constant([1,1,1,0,0,1,0],dtype=tf.float64)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y,logits = x)
with tf.Session() as sess:
    print(sess.run(loss))
