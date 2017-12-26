#_*_ coding:utf-8 _*_
#################################################################
#
#    file: variable_scope.py
#   usage: ./variable_scope.py
#   brief:
#  author: ***
#   email: ***
# created: 2017-12-26 11:42:38
#
#################################################################
#大大大部分情况下，跟 tf.get_variable() 配合使用，实现变量共享的功能。)
#为了更好地管理变量的命名空间而提出的。
import tensorflow as tf

def test_scope():
    with tf.name_scope('nsc1'):
        v1 = tf.Variable([1], name='v1')
        with tf.variable_scope('vsc1'):
            v2 = tf.Variable([1], name='v2')
            v3 = tf.get_variable(name='v3', shape=[])

    print("v1.name:%s" % v1.name)
    print("v2.name:%s" % v2.name)
    print("v3.name:%s" % v3.name)

'''
tf.name_scope() 并不会对 tf.get_variable() 创建的变量有任何影响。
tf.name_scope() 主要是用来管理命名空间的，这样子让我们的整个模型更加有条理。
tf.variable_scope() 的作用是为了实现变量共享，它和 tf.get_variable() 来完成变量共享的功能。
'''
#tf.Variable() 的方式来定义)
def my_image_filter():
    conv1_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),name="conv1_weights")
    conv1_biases = tf.Variable(tf.zeros([32]), name="conv1_biases")
    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 32]),name="conv2_weights")
    conv2_biases = tf.Variable(tf.zeros([32]), name="conv2_biases")
    return None
def test_variable():
    result1 = my_image_filter()
    result2 = my_image_filter()
    vs = tf.trainable_variables()
    for v in vs:
        print(v)

def conv_relu(kernel_shape, bias_shape):
    weights = tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.0))
    return None
def my_image_filter1():
    with tf.variable_scope("conv1"):
        relu1 = conv_relu([5, 5, 32, 32], [32])
    with tf.variable_scope("conv2"):
        return conv_relu( [5, 5, 32, 32], [32])

def test_get_variable():
    with tf.variable_scope("image_filters") as scope:
        result1 = my_image_filter1()
        scope.reuse_variables()
        result2 = my_image_filter1()
    vs = tf.trainable_variables()
    for v in vs:
        print(v)


#test_scope()
#test_variable()
test_get_variable()
