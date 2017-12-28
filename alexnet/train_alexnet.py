#_*_ coding:utf-8 _*_
from alexnet import AlexNet
import tensorflow as tf

X = tf.constant(0,shape=[64,227,227,3],dtype=tf.float32)
net = AlexNet(X,0.5,1000,1,1)
