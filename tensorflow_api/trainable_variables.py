##-*- coding:utf-8 -*-
#!/usr/bin/python
#################################################################
#
#    file: trainable_variables.py
#   usage: ./trainable_variables.py
#   brief:
#  author: ***
#   email: ***
# created: 2017-11-24 11:19:56
#
#################################################################
import tensorflow as tf

#tf.trainable_variables返回的是需要训练的变量列表
#tf.all_variables返回的是所有变量的列表
v = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='v')
v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')
global_step = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='global_step', trainable=False)
ema = tf.train.ExponentialMovingAverage(0.99, global_step)
for ele1 in tf.trainable_variables():
    print(ele1.name)
for ele2 in tf.global_variables():
    print(ele2.name)
