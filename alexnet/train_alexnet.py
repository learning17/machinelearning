#_*_ coding:utf-8 _*_
from alexnet import AlexNet
from generate_alexnet import ImageDataGenerator
import tensorflow as tf
import logging
import logging.handlers
handler = logging.handlers.RotatingFileHandler(
    "train_alexnet.log",
     maxBytes=5*1024*1024,
    backupCount=100,
    encoding='utf-8')
fmt = '%(asctime)s|%(message)s'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
logger = logging.getLogger("train_chatbot")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    with tf.device('/cpu:0'):
        hparams = tf.contrib.training.HParams(
            txt_file = "train.txt",
            mode = "training",
            batch_size = 64,
            num_classes = 2,
            shuffle = True)
        tr_data = ImageDataGenerator(hparams)
        tr_iterator = tr_data.get_iterator()

        hparams.txt_file = "val.txt"
        hparams.mode = "inference"
        hparams.shuffle = False
        val_data = ImageDataGenerator(hparams)
        val_iterator = val_data.get_iterator()

    X = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.float32,[None,hparams.num_classes])
    keep_prob = tf.placeholder(tf.float32)
    #learning_rate = tf.placeholder(tf.float32)
    learning_rate = tf.Variable(0.0, trainable=False)
    train_layers = ['fc8', 'fc7', 'fc6']

    model = AlexNet(X, keep_prob, hparams.num_classes,train_layers)
    y_ = model.create_model()
    loss, loss_summary = model.cross_entropy(y_, y)
    train_op, gradient_summary = model.optimizer(loss, learning_rate)
    accuracy, accuracy_summary = model.accuracy(y_, y)
    train_summary = tf.summary.merge(loss_summary + gradient_summary + accuracy_summary)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        #model initialize
        ckpt = tf.train.get_checkpoint_state('./ckpt')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
            model.initial_weights(sess)

        #data initialize
        sess.run(tf.tables_initializer())
        sess.run(tr_iterator.initializer)
        sess.run(val_iterator.initializer)
        sess.run(tf.assign(learning_rate, 0.02 * 0.97 ))
        summary_writer = tf.summary.FileWriter("train_alexnet", sess.graph)
        step = 0
        epoch = 0
        while True:
            #train
            step = model.global_step.eval()
            try:
                img_batch,labels_batch = sess.run([tr_iterator.img_batch,tr_iterator.labels_batch])
                _,acc,train_loss,step_summary = sess.run([train_op,accuracy, loss, train_summary],feed_dict={X:img_batch,
                                                                                           y:labels_batch,
                                                                                           keep_prob:0.5})
                logger.debug("train step:%d,loss:%f,acc:%f" % (step,train_loss,acc))
                summary_writer.add_summary(step_summary, step)
            except tf.errors.OutOfRangeError:
                epoch += 1
                sess.run(tf.assign(learning_rate, 0.02 * (0.97 ** epoch)))
                logger.debug("epoch:%d" % epoch)
                sess.run(tr_iterator.initializer)
                saver.save(sess,"./ckpt/alexnet.ckpt",global_step = step)
                #val
                val_acc = 0
                val_count = 0
                while True:
                    try:
                        img_batch,labels_batch = sess.run([val_iterator.img_batch,val_iterator.labels_batch])
                        acc = sess.run(accuracy, feed_dict={X:img_batch,
                                                            y:labels_batch,
                                                            keep_prob:1.0})
                        val_acc += acc
                        val_count += 1
                    except tf.errors.OutOfRangeError:
                        val_acc /= val_count
                        logger.debug("val Accuracy=%f" % val_acc)
                        sess.run(val_iterator.initializer)
                        break
                if val_acc > 0.97:
                    break

