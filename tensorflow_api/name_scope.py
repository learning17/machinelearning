#-*- coding:utf-8 -*-
import tensorflow as tf

with tf.name_scope('t1') as scope:
    y = tf.get_variable('y',[1]) #无视名称作用域的
    x = tf.Variable(1)

print(x.op.name)
print(y.op.name)
