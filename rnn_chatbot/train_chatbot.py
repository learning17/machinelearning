from generate_chatbot import Chatbot
from chatbot_model import chatbotModel
import tensorflow as tf
import numpy as np
import logging
import logging.handlers

if __name__ == '__main__':
    hparams = tf.contrib.training.HParams(
        batch_size = 50,
        share_vocab = True,
        eos = "</s>",
        sos = "<s>",
        unk = "<unk>",
        check_special_token = False,
        src_file = "train_encode",
        tgt_file = "train_decode",
        src_vocab_file = "vocab_encode",
        tgt_vocab_file = "vocab_decode",
        num_units = 256,
        num_layers = 4,
        mode = tf.contrib.learn.ModeKeys.TRAIN,
        attention_option = "luong",
        beam_width = 4)

    skip_count = tf.placeholder(tf.int64)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    chat = Chatbot(hparams)
    iterator = chat.get_iterator(skip_count)
    hparams.add_hparam("src_vocab_size",iterator.src_vocab_size)
    hparams.add_hparam("tgt_vocab_size",iterator.tgt_vocab_size)
    hparams.add_hparam("source_sequence_length",iterator.source_sequence_length)
    model = chatbotModel(hparams)
    logits, loss, final_context_state, sample_id = model.build_graph(iterator,keep_prob)
    learning_rate = tf.Variable(0.0, trainable=False)
    optimizer,gradient_norm_summary = model.optimizer_model(loss,learning_rate)
    saver = tf.train.Saver()
    train_summary = tf.summary.merge([
          tf.summary.scalar("loss",loss)] + gradient_norm_summary)
    handler = logging.handlers.RotatingFileHandler(
        "train_chatbot.log",
        maxBytes=5*1024*1024,
        backupCount=100,
        encoding='utf-8')
    fmt = '%(asctime)s|%(message)s'
    formatter = logging.Formatter(fmt)
    handler.setFormatter(formatter)
    logger = logging.getLogger("train_chatbot")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter("train_chatbot", sess.graph)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(tf.assign(learning_rate, 0.02 * 0.97 ))
        sess.run(iterator.initializer,feed_dict={skip_count:0})
        step = 0
        n = 0
        while True:
            try:
                _,train_loss,step_summary = sess.run([optimizer,loss,train_summary],feed_dict={keep_prob:0.5})
                logger.debug("step:%d,loss:%f" % (step,train_loss))
                #print("step:%d,loss:%f" % (step,train_loss))
                summary_writer.add_summary(step_summary, step)
                if step > 500000:
                    break
            except tf.errors.OutOfRangeError:
                n += 1
                sess.run(tf.assign(learning_rate, 0.02 * (0.97 ** n)))
                #print("out_range step:%d" % step)
                logger.debug("out_range step:%d" % step)
                sess.run(iterator.initializer,feed_dict={skip_count:0})
                continue
            step += 1
        saver.save(sess,"chatbot_model.ckpt")
        summary_writer.close()

