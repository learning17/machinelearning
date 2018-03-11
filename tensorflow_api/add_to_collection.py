import tensorflow as tf

v1 = tf.Variable(tf.constant(1))
v2 = tf.Variable(tf.constant(1))

name = "collection"
tf.add_to_collection(name,v1)
tf.add_to_collection(name,v2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
c1 = tf.get_collection(name)
print(sess.run(c1))

sess.run(tf.assign(v1,tf.constant(3)))
sess.run(tf.assign(v2,tf.constant(4)))

c2 = tf.get_collection(name)
print(sess.run(c2))

print(sess.run(tf.add_n(c2)))
