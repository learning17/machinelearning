import tensorflow as tf

global_step = tf.Variable(0, trainable=False)
boundaries = [10, 40]
values = [1.0, 0.5, 0.1]

learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for global_step in range(50):
        learning_rate = tf.train.piecewise_constant(global_step,boundaries,values)
        print(sess.run(learning_rate))
