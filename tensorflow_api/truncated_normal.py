#!/usr/bin/python
#################################################################
#
#    file: truncated_normal.py
#   usage: ./truncated_normal.py
#   brief:
#  author: *******
#   email: *******
# created: 2017-07-21 10:26:42
#
#################################################################
import tensorflow as tf
c = tf.truncated_normal(shape=[10,10], mean=0, stddev=1)
print tf.Session().run(c)
