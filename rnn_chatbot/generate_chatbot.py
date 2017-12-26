#-*- coding:utf-8 -*-
from tensorflow.python.ops import lookup_ops
import tensorflow as tf
import collections
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf8')

class BatchChatbot(
    collections.namedtuple("BatchChatbot",
                           ("initializer","source","target_input",
                            "target_output","source_sequence_length",
                            "target_sequence_length","src_vocab_size",
                            "tgt_vocab_size","src_vocab_table","tgt_vocab_table","reverse_tgt_vocab_table"))):
    pass

def check_vocab(vocab_file,check_special_token,sos,eos,unk):
    vocab = list()
    if tf.gfile.Exists(vocab_file):
        with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as f:
            for word in f:
                vocab.append(word.strip())
        if check_special_token:
            assert len(vocab) >= 3
            if vocab[0] != unk or vocab[1] != sos or vocab[2] != eos:
                vocab = [unk,sos,eos] + vocab
            with codecs.getwriter("utf-8")(tf.gfile.GFile(vocab_file, "wb")) as f:
                for word in vocab:
                    f.write("%s\n" % word)
    else:
        raise ValueError("vocab_file does not exist")
    return len(vocab)

def check_vocab_all(src_vocab_file,tgt_vocab_file,share_vocab,check_special_token,sos,eos,unk):
    src_vocab_size = check_vocab(src_vocab_file,check_special_token,sos,eos,unk)
    if share_vocab:
        tgt_vocab_size = src_vocab_size
    else:
        tgt_vocab_size = check_vocab(tgt_vocab_file,check_special_token,sos,eos,unk)
    return src_vocab_size,tgt_vocab_size

def create_vocab_tables(src_vocab_file,tgt_vocab_file,share_vocab):
    src_vocab_table = tf.contrib.lookup.index_table_from_file(src_vocab_file,default_value = 0)
    if share_vocab:
        tgt_vocab_table = src_vocab_table
    else:
        tgt_vocab_table = tf.contrib.lookup.index_table_from_file(tgt_vocab_file,default_value = 0)
    return src_vocab_table,tgt_vocab_table

class InferChatbot:
    def __init__(self,hparams):
        self.eos = hparams.eos
        self.sos = hparams.sos
        self.unk = hparams.unk
        self.src_vocab_size,self.tgt_vocab_size = check_vocab_all(
            hparams.src_vocab_file,
            hparams.tgt_vocab_file,
            hparams.share_vocab,
            hparams.check_special_token,
            self.sos,
            self.eos,
            self.unk)
        self.src_vocab_table,self.tgt_vocab_table = create_vocab_tables(
            hparams.src_vocab_file,
            hparams.tgt_vocab_file,
            hparams.share_vocab)
        self.reverse_tgt_vocab_table = self.create_index_vocab_tables(hparams.tgt_vocab_file)

    def create_index_vocab_tables(self,tgt_vocab_file):
        reverse_tgt_vocab_table = tf.contrib.lookup.index_to_string_table_from_file(tgt_vocab_file,default_value = self.unk)
        return reverse_tgt_vocab_table

    def get_infer_iterator(self,src_placeholder=None,batch_size_placeholder=None):
        src_dataset = tf.contrib.data.Dataset.from_tensor_slices(src_placeholder)
        src_eos_id = tf.cast(self.src_vocab_table.lookup(tf.constant(self.eos)),tf.int32)
        src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)
        src_dataset = src_dataset.map(
            lambda src: tf.cast(self.src_vocab_table.lookup(src), tf.int32))
        src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))
        def batching_func(src_tgt_dataset):
            return src_tgt_dataset.padded_batch(
                batch_size_placeholder,
                padded_shapes=(
                    tf.TensorShape([None]),
                    tf.TensorShape([])),
                padding_values=(
                    src_eos_id,
                    0))
        batched_dataset = batching_func(src_dataset)
        iterator = batched_dataset.make_initializable_iterator()
        src_ids, src_seq_len = iterator.get_next()
        return BatchChatbot(
            initializer = iterator.initializer,
            source = src_ids,
            target_input = None,
            target_output = None,
            source_sequence_length = src_seq_len,
            target_sequence_length = None,
            src_vocab_size = self.src_vocab_size,
            tgt_vocab_size = self.tgt_vocab_size,
            src_vocab_table = self.src_vocab_table,
            tgt_vocab_table = self.tgt_vocab_table,
            reverse_tgt_vocab_table = self.reverse_tgt_vocab_table)

class Chatbot:
    def __init__(self,hparams):
        self.batch_size = hparams.batch_size
        self.eos = hparams.eos
        self.sos = hparams.sos
        self.unk = hparams.unk
        self.src_dataset = tf.contrib.data.TextLineDataset(hparams.src_file)
        self.tgt_dataset = tf.contrib.data.TextLineDataset(hparams.tgt_file)
        self.src_vocab_size,self.tgt_vocab_size = check_vocab_all(
            hparams.src_vocab_file,
            hparams.tgt_vocab_file,
            hparams.share_vocab,
            hparams.check_special_token,
            self.sos,
            self.eos,
            self.unk)
        self.src_vocab_table,self.tgt_vocab_table = create_vocab_tables(
            hparams.src_vocab_file,
            hparams.tgt_vocab_file,
            hparams.share_vocab)
        self.num_threads = 4
        self.num_buckets = 4

    def get_iterator(self,skip_count=None):
        src_eos_id = tf.cast(self.src_vocab_table.lookup(tf.constant(self.eos)), tf.int32)
        tgt_sos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(self.sos)), tf.int32)
        tgt_eos_id = tf.cast(self.tgt_vocab_table.lookup(tf.constant(self.eos)), tf.int32)
        src_tgt_dataset = tf.contrib.data.Dataset.zip((self.src_dataset,self.tgt_dataset))
        #跳过前skip_count个样本
        if skip_count is not None:
            src_tgt_dataset = src_tgt_dataset.skip(skip_count)
        #选择output_buffer_size样本进行打乱，其他顺序不变
        output_buffer_size = 1000*self.batch_size
        #src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size,self.random_seed)
        src_tgt_dataset = src_tgt_dataset.shuffle(output_buffer_size)
        #将每个句子按照空格划分为当个字
        src_tgt_dataset = src_tgt_dataset.map(
                        lambda src, tgt:(tf.string_split([src]).values,
                                         tf.string_split([tgt]).values),
                        num_threads = self.num_threads,output_buffer_size = output_buffer_size)
        #过滤长度为0的句子
        src_tgt_dataset = src_tgt_dataset.filter(lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))
        #每个字转换为对应的数字ID
        src_tgt_dataset = src_tgt_dataset.map(
                        lambda src, tgt:(tf.cast(self.src_vocab_table.lookup(src), tf.int32),
                                         tf.cast(self.tgt_vocab_table.lookup(tgt), tf.int32)),
                        num_threads = self.num_threads,output_buffer_size = output_buffer_size)
        #tgt_input 开头添加 <s> tgt_output 结尾添加 </s>
        src_tgt_dataset = src_tgt_dataset.map(
                        lambda src, tgt: (src,
                                          tf.concat(([tgt_sos_id], tgt), 0),
                                          tf.concat((tgt, [tgt_eos_id]), 0)),
                        num_threads = self.num_threads,output_buffer_size = output_buffer_size)
        #计算src、tgt长度
        src_tgt_dataset = src_tgt_dataset.map(
                        lambda src, tgt_in, tgt_out:(src,tgt_in, tgt_out,
                                                     tf.size(src), tf.size(tgt_in)),
                        num_threads = self.num_threads,output_buffer_size = output_buffer_size)

        #批量返回
        def batching_func(src_tgt_dataset):
            return src_tgt_dataset.padded_batch(self.batch_size,
                                                padded_shapes=(
                                                    tf.TensorShape([None]),#src
                                                    tf.TensorShape([None]),#tgt_input
                                                    tf.TensorShape([None]),#tgt_output
                                                    tf.TensorShape([]),#src size
                                                    tf.TensorShape([])),#tgt size
                                                padding_values=(
                                                    src_eos_id,#src 用src_eos_id补齐
                                                    tgt_eos_id,#tgt_input 用tgt_eos_id补齐
                                                    tgt_eos_id,#tgt_output 用 tgt_eos_id补齐
                                                    0,
                                                    0))

        #按长度划分聚类
        if self.num_buckets > 1:
            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
                bucket_width = 10
                bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
                return tf.to_int64(tf.minimum(self.num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)
            batched_dataset = src_tgt_dataset.group_by_window(key_func = key_func,reduce_func=reduce_func,window_size=self.batch_size)
        else:
            batched_dataset = batching_func(src_tgt_dataset)
        iterator = batched_dataset.make_initializable_iterator()
        src_ids,tgt_input_ids,tgt_output_ids,src_seq_len,tgt_seq_len = iterator.get_next()
        return BatchChatbot(
            initializer = iterator.initializer,
            source = src_ids,
            target_input = tgt_input_ids,
            target_output = tgt_output_ids,
            source_sequence_length = src_seq_len,
            target_sequence_length = tgt_seq_len,
            src_vocab_size = self.src_vocab_size,
            tgt_vocab_size = self.tgt_vocab_size,
            src_vocab_table = self.src_vocab_table,
            tgt_vocab_table = self.tgt_vocab_table,
            reverse_tgt_vocab_table = None)
'''
#Chatbot
hparams = tf.contrib.training.HParams(
    src_file = "train_encode",
    tgt_file = "train_decode",
    src_vocab_file = "vocab",
    tgt_vocab_file = "vocab",
    eos = "</s>",
    sos = "<s>",
    unk = "<unk>",
    check_special_token = False,
    share_vocab = True,
    batch_size = 3)
c = Chatbot(hparams)
skip_count = tf.placeholder(tf.int64)
iterator = c.get_iterator(skip_count)

target_output = iterator.target_output
shape_size = tf.shape(target_output)
batch_size = tf.size(iterator.source_sequence_length)
with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(iterator.initializer,feed_dict={skip_count:0})
    #print sess.run(batch_size)
    print sess.run([target_output])
    #output_transpose = sess.run([tf.transpose(target_output)])
    #print output_transpose
    #print sess.run([shape_size])'''
'''
#InferChatbot
hparams = tf.contrib.training.HParams(
    src_vocab_file = "vocab_encode",
    tgt_vocab_file = "vocab_decode",
    eos = "</s>",
    sos = "<s>",
    unk = "<unk>",
    check_special_token = False,
    share_vocab = True)
inc = InferChatbot(hparams)
src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
iterator = inc.get_infer_iterator(src_placeholder,batch_size_placeholder)
input_data = ["吃 你 自 己 的 去"]
batch_size = 1
source = iterator.source
with tf.Session() as sess:
    tf.tables_initializer().run()
    sess.run(iterator.initializer,feed_dict={src_placeholder:input_data,batch_size_placeholder:batch_size})
    print sess.run(source)
'''
