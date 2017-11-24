import tensorflow as tf
a = tf.constant([1,2,3,4,5,6,2,3,1,1,2,3],shape=[1,3,4,1],dtype=tf.float32)
b = tf.constant([1,2,3,4,5,6,1,1,3],shape=[3,3,1,1],dtype=tf.float32)
c = tf.nn.conv2d(a,b,strides=[1, 2, 2, 1],padding='VALID')
d = tf.nn.conv2d(a,b,strides=[1, 2, 2, 1],padding='SAME')
e = tf.nn.conv2d_transpose(d,b,output_shape=[1,3,4,1],strides=[1, 2, 2, 1],padding='SAME')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))
