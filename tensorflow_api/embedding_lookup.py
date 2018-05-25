#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

c = np.random.random([10,1])
b = tf.nn.embedding_lookup(c,[1,3])#选取一个张量里面索引对应的元素

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(c)
    print('######################')
    print(sess.run(b))
