#!/usr/bin/python
#################################################################
#
#    file: sparse_softmax_cross_entropy_with_logits.py
#   usage: ./sparse_softmax_cross_entropy_with_logits.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-26 21:19:10
#
#################################################################
import tensorflow as tf

logits = tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])
y = tf.nn.softmax(logits)
with tf.Session() as sess:
    print sess.run(y)
