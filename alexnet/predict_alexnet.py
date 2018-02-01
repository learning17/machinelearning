#_*_ coding:utf-8 _*_
from alexnet import AlexNet
from generate_alexnet import ImageDataGenerator
from cat_dog_classes import class_names as cat_dog_class_names
import tensorflow as tf
import numpy as np
import sys

def predict_imagenet(filename):
    test_x = ImageDataGenerator.get_image(filename)
    X = tf.placeholder(tf.float32, [None, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)
    model = AlexNet(X,keep_prob)
    y_ = model.create_model()
    prob = tf.nn.softmax(y_)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.initial_weights(sess)
        y_,output =  sess.run([y_,prob],feed_dict={X: [test_x], keep_prob: 1})
        ImageDataGenerator.predict(output)

def predict_dog_cat(filename):
    test_x = ImageDataGenerator.get_image(filename)
    X = tf.placeholder(tf.float32, [None, 227, 227, 3])
    keep_prob = tf.placeholder(tf.float32)
    model = AlexNet(X, keep_prob, 2)
    y_ = model.create_model()
    predict = tf.argmax(y_,1)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state('./ckpt')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("no model")
            return
        pred = sess.run(predict,feed_dict={X: [test_x], keep_prob: 1})
        print(cat_dog_class_names[pred[0]])

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Useage:python type imageName")
    else:
        if sys.argv[1] == '1':
            predict_imagenet(sys.argv[2])
        else:
            predict_dog_cat(sys.argv[2])

