import tensorflow as tf

a = tf.get_variable("a",[2,2],initializer=tf.random_normal_initializer())
b = tf.get_variable("b",[3,3],initializer=tf.random_normal_initializer())
tf.add_to_collection("collection",a)
tf.add_to_collection("collection",b)
c = tf.get_collection("collection")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for k in c:
        print(sess.run(k))
