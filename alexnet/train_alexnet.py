#_*_ coding:utf-8 _*_
from alexnet import AlexNet
from generate_alexnet import ImageDataGenerator
import tensorflow as tf

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        hparams = tf.contrib.training.HParams(
            txt_file = "train.txt",
            mode = "training",
            batch_size = 64,
            num_classes = 2,
            shuffle = True)
        tr_data = ImageDataGenerator(hparams)

        hparams.txt_file = "val.txt"
        hparams.mode = "inference"
        hparams.shuffle = False
        tr_data = ImageDataGenerator(hparams)

    X = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.float32,[None,hparams.num_classes])
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_layers = ['fc8', 'fc7', 'fc6']

    model = AlexNet(X, keep_prob, hparams.num_classes,train_layers)
    y_ = model.create_model()
    loss, loss_summary = model.cross_entropy(y_, y)
    train_op, gradient_summary = model.optimizer(loss, learning_rate)
    accuracy, accuracy_summary = model.accuracy(y_, y)
    train_summary = tf.summary.merge(loss_summary + gradient_summary + accuracy_summary)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(tf.assign(learning_rate, 0.02 * 0.97 ))

        summary_writer = tf.summary.FileWriter("train_chatbot", sess.graph)



