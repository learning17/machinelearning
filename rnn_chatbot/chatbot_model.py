from generate_chatbot import *
from tensorflow.python.layers import core as layers_core
import tensorflow as tf

class chatbotModel:
    def __init__(self,hparams):
        self.eos = hparams.eos
        self.sos = hparams.sos
        self.unk = hparams.unk
        self.num_units = hparams.num_units
        self.num_layers = hparams.num_layers
        self.src_vocab_size = hparams.src_vocab_size
        self.tgt_vocab_size = hparams.tgt_vocab_size
        self.mode = hparams.mode
        self.beam_width = hparams.beam_width
        self.attention_option = hparams.attention_option
        self.embedding_encoder,self.embedding_decoder = self.init_embeddings(hparams.share_vocab)
        self.batch_size = tf.size(hparams.source_sequence_length)
        with tf.variable_scope("build_network"):
            with tf.variable_scope("decoder/output_projection"):
                self.output_layer = layers_core.Dense(self.tgt_vocab_size,use_bias=False)

    def init_embeddings(self,share_vocab):
        with tf.variable_scope('embedding'):
            if share_vocab:
                if self.src_vocab_size != self.tgt_vocab_size:
                    raise ValueError("share_vocab but src_vocab_size not equal to tgt_vocab_size")
                embedding = tf.get_variable('embedding_share', [self.src_vocab_size,self.num_units])
                embedding_encoder = embedding
                embedding_decoder = embedding
            else:
                with tf.variable_scope("encoder"):
                    embedding_encoder = tf.get_variable('embedding_encoder', [self.src_vocab_size,self.num_units])
                with tf.variable_scope("decoder"):
                    embedding_decoder = tf.get_variable("embedding_decoder", [self.tgt_vocab_size,self.num_units])
        return embedding_encoder,embedding_decoder

    def _single_cell_fn(self,keep_prob):
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units = self.num_units,state_is_tuple = True)
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            single_cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob = keep_prob)
        return single_cell

    def _create_rnn_cell(self,keep_prob):
        cell_list = list()
        for i in range(self.num_layers):
            single_cell = self._single_cell_fn(keep_prob)
            cell_list.append(single_cell)
        cell = tf.contrib.rnn.MultiRNNCell(cell_list,state_is_tuple = True)
        return cell

    def _build_encoder(self,iterator,keep_prob):
        with tf.variable_scope("encoder") as scope:
            dtype = scope.dtype
            encoder_emb_inp = tf.nn.embedding_lookup(self.embedding_encoder,iterator.source)
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                encoder_emb_inp = tf.nn.dropout(encoder_emb_inp,keep_prob)
            cell = self._create_rnn_cell(keep_prob)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell,encoder_emb_inp,dtype=dtype,
                    sequence_length = iterator.source_sequence_length)
        return encoder_outputs,encoder_state

    def create_attention_mechanism(self,memory,source_sequence_length):
        if self.attention_option == "luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.num_units,
                memory,
                memory_sequence_length=source_sequence_length)
        elif self.attention_option == "scaled_luong":
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self.num_units,
                memory,
                memory_sequence_length=source_sequence_length,
                scale=True)
        elif self.attention_option == "bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.num_units,
                memory,
                memory_sequence_length=source_sequence_length)
        elif self.attention_option == "normed_bahdanau":
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                self.num_units,
                memory,
                memory_sequence_length=source_sequence_length,
                normalize=True)
        else:
            raise ValueError("Unknown attention option %s" % attention_option)
        return attention_mechanism

    def _build_decoder_cell(self,encoder_outputs,encoder_state,source_sequence_length,keep_prob):
        memory = encoder_outputs
        if self.mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width > 0:
            memory = tf.contrib.seq2seq.tile_batch(memory,multiplier = self.beam_width)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length,multiplier=self.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state,multiplier=self.beam_width)
            batch_size = self.batch_size * self.beam_width
        else:
            batch_size = self.batch_size
        #attention_mechanism
        attention_mechanism = self.create_attention_mechanism(memory,source_sequence_length)
        cell = self._create_rnn_cell(keep_prob)
        cell = tf.contrib.seq2seq.AttentionWrapper(
                cell,
                attention_mechanism,
                attention_layer_size = self.num_units,
                alignment_history = False)
        decoder_initial_state = cell.zero_state(batch_size,tf.float32).clone(cell_state = encoder_state)
        return cell,decoder_initial_state

    def _build_decoder(self,iterator,encoder_outputs,encoder_state,keep_prob):
        tgt_sos_id = tf.cast(iterator.tgt_vocab_table.lookup(tf.constant(self.sos)),tf.int32)
        tgt_eos_id = tf.cast(iterator.tgt_vocab_table.lookup(tf.constant(self.eos)),tf.int32)
        maximum_iterations = tf.to_int32(tf.round(tf.to_float(tf.reduce_max(iterator.source_sequence_length))* 20.0))
        with tf.variable_scope("decoder") as decoder_scope:
            cell,decoder_initial_state = self._build_decoder_cell(encoder_outputs,
                                                                  encoder_state,
                                                                  iterator.source_sequence_length,
                                                                  keep_prob)
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder,iterator.target_input)
                helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp,iterator.target_sequence_length)
                my_decoder = tf.contrib.seq2seq.BasicDecoder(cell,helper,decoder_initial_state)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                                                    my_decoder,
                                                    swap_memory=True,
                                                    scope=decoder_scope)
                sample_id = outputs.sample_id
                logits = self.output_layer(outputs.rnn_output)
            else:
                start_tokens = tf.fill([self.batch_size], tgt_sos_id)
                end_token = tgt_eos_id
                if self.beam_width > 0:
                    my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                                    cell=cell,
                                    embedding=self.embedding_decoder,
                                    start_tokens=start_tokens,
                                    end_token=end_token,
                                    initial_state=decoder_initial_state,
                                    beam_width=self.beam_width,
                                    output_layer=self.output_layer)
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                                self.embedding_decoder,
                                start_tokens,
                                end_token)
                    my_decoder = tf.contrib.seq2seq.BasicDecoder(
                                cell,
                                helper,
                                decoder_initial_state,
                                output_layer=self.output_layer)
                outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
                                                    my_decoder,
                                                    maximum_iterations=maximum_iterations,
                                                    swap_memory=True,
                                                    scope=decoder_scope)
                if self.beam_width > 0:
                    logits = tf.no_op()
                    sample_id = outputs.predicted_ids
                else:
                    logits = outputs.rnn_output
                    sample_id = outputs.sample_id
        return logits,sample_id,final_context_state

    def loss_model(self,target_output,logits,target_sequence_length):
        max_time = tf.shape(target_output)[1]
        cross = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(target_sequence_length,max_time,dtype=logits.dtype)
        loss = tf.reduce_sum(cross * target_weights)/tf.to_float(self.batch_size)
        return loss

    def build_graph(self,iterator,keep_prob):
        with tf.variable_scope("dynamic_seq2seq"):
            encoder_outputs, encoder_state = self._build_encoder(iterator,keep_prob)
            logits, sample_id, final_context_state = self._build_decoder(iterator,encoder_outputs,encoder_state,keep_prob)
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                loss = self.loss_model(iterator.target_output,logits,iterator.target_sequence_length)
            else:
                loss = None
        return logits, loss, final_context_state, sample_id

    def optimizer_model(self,loss,learning_rate):
        tvars = tf.trainable_variables()
        clipped_gradient, gradient_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        gradient_norm_summary = [tf.summary.scalar("gradient_norm", gradient_norm)]
        gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradient)))
        gradient_norm_summary.append(tf.summary.scalar("learning_rate",learning_rate))
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(clipped_gradient, tvars))
        return optimizer,gradient_norm_summary

