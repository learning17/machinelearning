#-*- coding:utf-8 -*-
from tensorflow.python.training import moving_averages
import tensorflow as tf

moving_mean = tf.get_variable("moving_mean",shape = [2,3], dtype=tf.float32, initializer=tf.zeros_initializer,trainable=False)
moving_variance = tf.get_variable("moving_variance",shape = [2,3], dtype=tf.float32, initializer=tf.zeros_initializer,trainable=False)

mean = tf.constant([1,2,3,4,5,6],shape = [2,3],dtype=tf.float32)
#给图中的某些计算指定顺序,先执行assign_moving_average，然后再执行moving_mean
with tf.control_dependencies([moving_averages.assign_moving_average(moving_mean,mean,0.9997)]):
    moving_mean = tf.identity(moving_mean)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(moving_mean))
    print(sess.run(moving_variance))

