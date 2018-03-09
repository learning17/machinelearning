import tensorflow as tf

l2_reg = tf.contrib.layers.l2_regularizer(0.1)
tmp = tf.constant([0,1,2,3],dtype=tf.float32)
collections = [tf.GraphKeys.GLOBAL_VARIABLES,"test"]
a=tf.get_variable("I_am_a",
                  regularizer=l2_reg,
                  initializer=tmp,
                  collections=collections)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))
    reg_set=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)#REGULARIAZTION_LOSSES集合会包含所有被weight_decay后的参数和
    print(sess.run(reg_set))
    keys = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    for key in keys:
        print(key.name)
    keys = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)#regularizer定义会将a加入REGULARIZATION_LOSSES集合
    for key in keys:
        print(key.name)
    keys = tf.get_collection("test")
    for key in keys:
        print(key.name)


