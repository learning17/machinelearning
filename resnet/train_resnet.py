#-*- coding:utf-8 -*-
from generate_resnet import ImageNetGenerator
from resnet import Resnet
import tensorflow as tf
import logging
import logging.handlers

handler = logging.handlers.RotatingFileHandler(
    "train_resnet.log",
    maxBytes=5*1024*1024,
    backupCount=100,
    encoding='utf-8')
formatter = logging.Formatter('%(asctime)s|%(message)s')
handler.setFormatter(formatter)
log = logging.getLogger("resnet")
log.addHandler(handler)
log.setLevel(logging.DEBUG)

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        hparams = tf.contrib.training.HParams(
            data_dir = "/data/ImageNet/ILSVRC2012_img_train",
            txt_file = "train.txt",
            is_training = False,
            num_classes = 1000,
            batch_size = 256,
            shuffle = True,
            height = 224,
            width = 224)
        tr_data = ImageNetGenerator(hparams)
        tr_iterator = tr_data.get_iterator()

        hparams.txt_file = "dev.txt"
        hparams.is_training = False
        hparams.shuffle = False
        dev_data = ImageNetGenerator(hparams)
        dev_iterator = dev_data.get_iterator()

    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32,[None,hparams.num_classes])
    is_training = tf.placeholder(tf.bool)
    learning_rate = tf.Variable(0.0, trainable=False)
    model = Resnet(is_training)
    y_ = model.create_model(X)
    loss, loss_summary = model.cross_entropy(y_, y)
    train_op, gradient_summary = model.optimizer(loss, learning_rate)
    accuracy, accuracy_summary = model.accuracy(y_, y)
    train_summary = tf.summary.merge(loss_summary + gradient_summary + accuracy_summary)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./ckpt')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(tr_iterator.initializer)
        sess.run(dev_iterator.initializer)
        sess.run(tf.assign(learning_rate, 0.1))
        summary_writer = tf.summary.FileWriter("train_resnet", sess.graph)
        step = 0
        epoch = 0
        while True:
            step = model.global_step.eval()
            try:
                img_batch,labels_batch = sess.run([tr_iterator.img_batch,tr_iterator.labels_batch])
                _,acc,train_loss,step_summary = sess.run([train_op,accuracy, loss, train_summary],feed_dict={X:img_batch,
                                                                                                         y:labels_batch,
                                                                                                         is_training:True})
                log.debug("train step:%d,loss:%f,acc:%f" % (step,train_loss,acc))
                summary_writer.add_summary(step_summary, step)
            except tf.errors.OutOfRangeError:
                epoch += 1
                #n = epoch/10
                sess.run(tf.assign(learning_rate, 0.1 * (0.9 ** n)))
                log.debug("epoch:%d" % epoch)
                sess.run(tr_iterator.initializer)
                saver.save(sess,"./ckpt/resnet.ckpt",global_step = step)
                dev_acc = 0
                dev_count = 0
                while True:
                    try:
                        img_batch,labels_batch = sess.run([dev_iterator.img_batch,dev_iterator.labels_batch])
                        acc = sess.run(accuracy, feed_dict={X:img_batch,
                                                            y:labels_batch,
                                                            is_training:False})
                        dev_acc += acc
                        dev_count += 1
                    except tf.errors.OutOfRangeError:
                        dev_acc /= dev_count
                        log.debug("dev Accuracy=%f" % dev_acc)
                        sess.run(dev_iterator.initializer)
                        break
                if dev_acc > 0.999:
                    break

