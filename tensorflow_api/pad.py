#-*- coding:utf-8 -*-
import tensorflow as tf

t = tf.constant([[1, 2, 3], [4, 5, 6]])
paddings = tf.constant([[1, 2,], [2, 2]]) #padding是一个[n,2]的tensor，n是t的秩，例如[1,2]表示在维度1的前面补齐1个0，后面补齐2个0
p = tf.pad(t, paddings, "CONSTANT")
with tf.Session() as sess:
    print(sess.run(p))
