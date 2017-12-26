#-*- coding:utf-8 -*-
#!/usr/bin/python
#################################################################
#
#    file: get_shape.py
#   usage: ./get_shape.py
#   brief:
#  author: ***
#   email: ***
# created: 2017-11-24 16:36:59
#
#################################################################
import tensorflow as tf
import numpy as np

#tf.shape(a)和a.get_shape()比较
#相同点：都可以得到tensor a的尺寸
#不同点：tf.shape()中a 数据的类型可以是tensor, list, array
#a.get_shape()中a的数据类型只能是tensor,且返回的是一个元组

x=tf.constant([[1,2,3],[4,5,6]])
y = [[1,2,3],[4,5,6]]
z=np.arange(24).reshape([2,3,4])
x_shape=tf.shape(x)
y_shape=tf.shape(y)
z_shape=tf.shape(z)
with tf.Session() as sess:
    print(sess.run(x_shape))
    print(sess.run(y_shape))
    print(sess.run(z_shape))

print(x.get_shape())#不能使用 sess.run() 因为返回的不是tensor 或string,而是元组
print(x.get_shape().as_list())
print(y.get_shape())
print(z.get_shape())
