#-*- coding:utf-8 -*-
import tensorflow as tf

#1)tensorflow 提供conv2d 的"SAME"方式,先向右滑动卷积，如果发现最右侧不够和卷积核卷积，进行填充，然后卷积；
#2）如果想要实现SAME，可以先左右填充，然后再采用VALID方式卷积

def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                        [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    print(inputs.shape)
    return tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
        padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
        kernel_initializer=tf.ones_initializer,
        data_format=data_format)
initial = tf.truncated_normal(shape=[1,4,4,1],mean=0,stddev=1)
a=tf.Variable(initial)
with tf.variable_scope('A') as scope:
    b = tf.layers.conv2d(inputs=a,filters=1,kernel_size=3,strides=2,padding='SAME',
                     use_bias=False,kernel_initializer=tf.ones_initializer,data_format="channels_last",name="test")
c = conv2d_fixed_padding(a,1,3,2,'channels_last')
f = tf.constant(1,shape=[3,3,1,1],dtype=tf.float32)
d = tf.nn.conv2d(a,f,[1,2,2,1],"SAME")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print("###################")
    print(sess.run(b))
    print("####################")
    print(sess.run(c))
    print("######################")
    print(sess.run(d))
vs = tf.trainable_variables()
for v in vs:
    print(v)
