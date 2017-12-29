from alexnet import AlexNet
import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 227, 227, 3])
keep_prob = tf.placeholder(tf.float32)
model = AlexNet(X,keep_prob)
train_layers = ['fc8', 'fc7', 'fc6']
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]
print(var_list)
