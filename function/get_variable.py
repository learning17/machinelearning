#!/usr/bin/python
#################################################################
#
#    file: get_variable.py
#   usage: ./get_variable.py
#   brief:
#  author: ***
#   email: ***
# created: 2017-11-24 17:36:15
#
#################################################################
import tensorflow as tf

def conv_relu(kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.random_normal_initializer())
    return None

def my_image_filter():
    with tf.variable_scope("conv1"):#Variables created here will be named "conv1/weights", "conv1/biases".
        relu1 = conv_relu([5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):#Variables created here will be named "conv2/weights", "conv2/biases".
        return conv_relu([5, 5, 32, 32], [32])

with tf.variable_scope("image_filters") as scope:
    result1 = my_image_filter()
    scope.reuse_variables()#共享变量，否则出现variable already exists
    result2 = my_image_filter()
vs = tf.trainable_variables()
for v in vs:
    print(v)
