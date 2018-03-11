#-*- coding:utf-8 -*-
import tensorflow as tf
'''
moments函数就是在 [0] 维度上求了个均值和方差,[0]维度方向挤压。
针对2×3大小的矩阵，axis还可以这么理解，若axis = [0]，那么我们2×3的小矩阵
可以理解成是一个包含了2个长度为3的一维向量，然后就是求这两个向量的均值和方差
'''
img = tf.Variable(tf.random_normal([2,3]))
axis = list(range(len(img.get_shape())-1))
mean, variance = tf.nn.moments(img,axis)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(img))
    print(sess.run(mean))
    print(sess.run(variance))
