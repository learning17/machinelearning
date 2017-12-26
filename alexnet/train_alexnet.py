#_*_ coding:utf-8 _*_
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import alexnet
import tensorflow as tf

with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    inputs = tf.constant(0,shape=[50,224,224,3],dtype=tf.float32)
    outputs, end_points = alexnet.alexnet_v2(inputs)
    print(end_points)
