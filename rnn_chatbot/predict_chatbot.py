#-*- coding:utf-8 -*-
from generate_chatbot import *
from chatbot_model import chatbotModel
import tensorflow as tf
import numpy as np
import sys

def main(argv):
    input_data = argv.split("|")
    #input_data = ["亲 亲 亲"]
    input_batch_size = len(input_data)
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
            num_units = 128,
            num_layers = 2,
            mode = tf.contrib.learn.ModeKeys.INFER,
            attention_option = "luong",
            beam_width = 4)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
    batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
    infer_chat = InferChatbot(hparams)
    iterator = infer_chat.get_infer_iterator(src_placeholder,batch_size_placeholder)
    hparams.add_hparam("src_vocab_size",iterator.src_vocab_size)
    hparams.add_hparam("tgt_vocab_size",iterator.tgt_vocab_size)
    hparams.add_hparam("source_sequence_length",iterator.source_sequence_length)
    infer_model = chatbotModel(hparams)
    logits,_,final_context_state,sample_id = infer_model.build_graph(iterator,keep_prob)
    sample_words = tf.transpose(iterator.reverse_tgt_vocab_table.lookup(tf.to_int64(sample_id)))
    saver = tf.train.Saver()

    def get_result(infer_sample_words):
        num_sentence = infer_sample_words.shape[0]
        num_batch = infer_sample_words.shape[2]
        for i in range(num_batch):
            print("ask:%s" % input_data[i])
            print("answer top %d:" % num_sentence)
            for j in range(num_sentence):
                sentence_list = infer_sample_words[j,:,i].tolist()
                if hparams.eos and hparams.eos in sentence_list:
                    sentence_list = sentence_list[:sentence_list.index(hparams.eos)]
                sentence = " ".join(sentence_list)
                print("  " + sentence)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        sess.run(iterator.initializer,feed_dict={src_placeholder:input_data,batch_size_placeholder:input_batch_size})
        saver.restore(sess,"chatbot_model.ckpt")
        while True:
            try:
                _,infer_sample_words = sess.run([logits,sample_words])
                if hparams.beam_width == 0:
                    infer_sample_words = np.expand_dims(infer_sample_words,0)
                get_result(infer_sample_words)
            except tf.errors.OutOfRangeError:
                print("done")
                break
if __name__ == '__main__':
    main(sys.argv[1])
