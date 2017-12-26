TensorFlow - 基于 RNN 训练聊天机器人
============================

## 简介

基于 TensoFlow 构建 SeqSeq 模型，并加入 Attention 机制，encoder 和 decoder 为 3 层的 RNN 网络。本教程主要参考 Tensorflow 官网 translate Demo。

### 步骤简介
本教程一共分为四个部分
* `generate_chat.py` - 清洗数据、提取 ask 数据和 answer 数据、提取词典、为每个字生成唯一的数字 ID、ask 和 answer 用数字 ID 表示；
* `seq2seq.py、seq2seq_model.py` - Tensorflow 中 Translate Demo，由于出现 deepcopy 错误，这里对 SeqSeq 稍微改动了；
* `train_chat.py` - 训练 SeqSeq 模型；
* `predict_chat.py` - 进行聊天。

## 数据学习

### 获取训练数据
我们在腾讯云的 COS 上准备了训练数据，使用 `wget` 命令获取：
```
wget http://tensorflow-1253902462.cosgz.myqcloud.com/rnn_chat/chat.conv
```
> <checker type="output-contains" command="ls /home/ubuntu/chat.conv" hint="请使用 wget 下载训练数据">
>   <keyword regex="/home/ubuntu/chat.conv" />
> </checker>
### 数据预处理

#### 处理思路：
* 原始数据中，每次对话是 M 开头，前一行是 E ，并且每次对话都是一问一答的形式。将原始数据分为 ask、answer 两份数据；
* 两种词袋：“汉字 => 数字”、“数字 => 汉字”，根据第一个词袋将 ask、answer 数据转化为数字表示；
* answer 数据每句添加 EOS 作为结束符号。

#### 示例代码：
现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 generate_chat.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 generate_chat.py" />

```python
/// <example verb="create" file="/home/ubuntu/generate_chat.py" />
#-*- coding:utf-8 -*-
from io import open
import random
import sys
import tensorflow as tf

PAD = "PAD"
GO = "GO"
EOS = "EOS"
UNK = "UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0 #填充
GO_ID = 1  #开始标志
EOS_ID = 2 #结束标志
UNK_ID = 3 #未知字符
_buckets = [(10, 15), (20, 25), (40, 50),(80,100)]
units_num = 256
num_layers = 3
max_gradient_norm = 5.0
batch_size = 50
learning_rate = 0.5
learning_rate_decay_factor = 0.97

train_encode_file = "train_encode"
train_decode_file = "train_decode"
test_encode_file = "test_encode"
test_decode_file = "test_decode"
vocab_encode_file = "vocab_encode"
vocab_decode_file = "vocab_decode"
train_encode_vec_file = "train_encode_vec"
train_decode_vec_file = "train_decode_vec"
test_encode_vec_file = "test_encode_vec"
test_decode_vec_file = "test_decode_vec"

def is_chinese(sentence):
    flag = True
    if len(sentence) < 2:
        flag = False
        return flag
    for uchar in sentence:
        if(uchar == '，' or uchar == '。' or
            uchar == '～' or uchar == '?' or
            uchar == '！'):
            flag = True
        elif '\u4e00' <= uchar <= '\u9fff':
            flag = True
        else:
            flag = False
            break
    return flag

def get_chatbot():
    f = open("chat.conv","r", encoding="utf-8")
    train_encode = open(train_encode_file,"w", encoding="utf-8")
    train_decode = open(train_decode_file,"w", encoding="utf-8")
    test_encode = open(test_encode_file,"w", encoding="utf-8")
    test_decode = open(test_decode_file,"w", encoding="utf-8")
    vocab_encode = open(vocab_encode_file,"w", encoding="utf-8")
    vocab_decode = open(vocab_decode_file,"w", encoding="utf-8")
    encode = list()
    decode = list()

    chat = list()
	print("start load source data...")
	step = 0
    for line in f.readlines():
        line = line.strip('\\n').strip()
        if not line:
            continue
        if line[0] == "E":
			if step % 1000 == 0:
				print("step:%d" % step)
			step += 1
            if(len(chat) == 2 and is_chinese(chat[0]) and is_chinese(chat[1]) and
                not chat[0] in encode and not chat[1] in decode):
                encode.append(chat[0])
                decode.append(chat[1])
            chat = list()
        elif line[0] == "M":
            L = line.split(' ')
            if len(L) > 1:
                chat.append(L[1])
    encode_size = len(encode)
    if encode_size != len(decode):
        raise ValueError("encode size not equal to decode size")
    test_index = random.sample([i for i in range(encode_size)],int(encode_size*0.2))
	print("divide source into two...")
	step = 0
    for i in range(encode_size):
		if step % 1000 == 0:
			print("%d" % step)
		step += 1
        if i in test_index:
            test_encode.write(encode[i] + "\\n")
            test_decode.write(decode[i] + "\\n")
        else:
            train_encode.write(encode[i] + "\\n")
            train_decode.write(decode[i] + "\\n")

    vocab_encode_set = set(''.join(encode))
    vocab_decode_set = set(''.join(decode))
	print("get vocab_encode...")
	step = 0
    for word in vocab_encode_set:
		if step % 1000 == 0:
			print("%d" % step)
		step += 1
        vocab_encode.write(word + "\\n")
	print("get vocab_decode...")
	step = 0
    for word in vocab_decode_set:
		print("%d" % step)
		step += 1
        vocab_decode.write(word + "\\n")

def gen_chatbot_vectors(input_file,vocab_file,output_file):
    vocab_f = open(vocab_file,"r", encoding="utf-8")
    output_f = open(output_file,"w")
    input_f = open(input_file,"r",encoding="utf-8")
    words = list()
    for word in vocab_f.readlines():
        word = word.strip('\\n').strip()
        words.append(word)
    word_to_id = {word:i for i,word in enumerate(words)}
    to_id = lambda word: word_to_id.get(word,UNK_ID)
	print("get %s vectors" % input_file)
	step = 0
    for line in input_f.readlines():
		if step % 1000 == 0:
			print("step:%d" % step)
		step += 1
        line = line.strip('\\n').strip()
        vec = map(to_id,line)
        output_f.write(' '.join([str(n) for n in vec]) + "\\n")

def get_vectors():
    gen_chatbot_vectors(train_encode_file,vocab_encode_file,train_encode_vec_file)
    gen_chatbot_vectors(train_decode_file,vocab_decode_file,train_decode_vec_file)
    gen_chatbot_vectors(test_encode_file,vocab_encode_file,test_encode_vec_file)
    gen_chatbot_vectors(test_decode_file,vocab_decode_file,test_decode_vec_file)

def get_vocabs(vocab_file):
    words = list()
    with open(vocab_file,"r", encoding="utf-8") as vocab_f:
        for word in vocab_f:
            words.append(word.strip('\\n').strip())
    id_to_word = {i: word for i, word in enumerate(words)}
    word_to_id = {v: k for k, v in id_to_word.items()}
    vocab_size = len(id_to_word)
    return id_to_word,word_to_id,vocab_size

def read_data(source_path, target_path, max_size=None):
    data_set = [[] for _ in _buckets]
    with tf.gfile.GFile(source_path, mode="r") as source_file:
        with tf.gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(_buckets):
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                        data_set[bucket_id].append([source_ids, target_ids])
                        break
                source, target = source_file.readline(), target_file.readline()
    return data_set

```
> <checker type="output-contains" command="ls /home/ubuntu/generate_chat.py" hint="请创建并执行 /home/ubuntu/generate_chat.py">
>    <keyword regex="/home/ubuntu/generate_chat.py" />
> </checker>

#### 生成数据：
可以在终端中一步一步执行下面命令

启动 python：
```python
cd /home/ubuntu/
python
from generate_chat import *
```
获取 ask、answer 数据并生成字典：
```python
get_chatbot()
```

* [train_encode][output-1-1] - 用于训练的 ask 数据；

> <edit for="output-1-1" file="/home/ubuntu/train_encode" />

* [train_decode][output-1-2] - 用于训练的 answer 数据；

> <edit for="output-1-2" file="/home/ubuntu/train_decode" />

* [test_encode][output-1-3] - 用于验证的 ask 数据；

> <edit for="output-1-3" file="/home/ubuntu/test_encode" />


* [test_decode][output-1-4] - 用于验证的 answer 数据；

> <edit for="output-1-4" file="/home/ubuntu/test_decode" />

* [vocab_encode][output-1-5] - ask 数据词典；

> <edit for="output-1-5" file="/home/ubuntu/vocab_encode" />

* [vocab_decode][output-1-6] - answer 数据词典。

> <edit for="output-1-6" file="/home/ubuntu/vocab_decode" />

> <checker type="output-contains" command="ls /home/ubuntu/train_encode" hint="执行 get_chatbot()">
>    <keyword regex="/home/ubuntu/train_encode" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/train_decode" hint="执行 get_chatbot()">
>    <keyword regex="/home/ubuntu/train_decode" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/test_encode" hint="执行 get_chatbot()">
>    <keyword regex="/home/ubuntu/test_encode" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/test_decode" hint="执行 get_chatbot()">
>    <keyword regex="/home/ubuntu/test_decode" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/vocab_encode" hint="执行 get_chatbot()">
>    <keyword regex="/home/ubuntu/vocab_encode" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/vocab_decode" hint="执行 get_chatbot()">
>    <keyword regex="/home/ubuntu/vocab_decode" />
> </checker>

训练数据转化为数字表示：
```python
get_vectors()
```
* [train_encode_vec][output-2-1] - 用于训练的 ask 数据数字表示形式；

> <edit for="output-2-1" file="/home/ubuntu/train_encode_vec" />

* [train_decode_vec][output-2-2] - 用于训练的 answer 数据数字表示形式；

> <edit for="output-2-2" file="/home/ubuntu/train_decode_vec" />

* [test_encode_vec][output-2-3] - 用于验证的 ask 数据；

> <edit for="output-2-3" file="/home/ubuntu/test_encode_vec" />

* [test_decode_vec][output-2-4] - 用于验证的 answer 数据；

> <edit for="output-2-4" file="/home/ubuntu/test_decode_vec" />

> <checker type="output-contains" command="ls /home/ubuntu/train_encode_vec" hint="执行 get_vectors()">
>    <keyword regex="/home/ubuntu/train_encode_vec" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/train_decode_vec" hint="执行 get_vectors()">
>    <keyword regex="/home/ubuntu/train_decode_vec" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/test_encode_vec" hint="执行 get_vectors()">
>    <keyword regex="/home/ubuntu/test_encode_vec" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/test_decode_vec" hint="执行 get_vectors()">
>    <keyword regex="/home/ubuntu/test_decode_vec" />
> </checker>

## 模型学习

### Seq2Seq 模型
采用 translate 的model，实验过程发现 deepcopy 出现 NotImplementedType 错误，所以对 translate 中 seq2seq 稍微改动了。
#### seq2seq示例代码：
现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 seq2seq.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 seq2seq.py" />

```python
/// <example verb="create" file="/home/ubuntu/seq2seq.py" />
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Library for creating sequence-to-sequence models in TensorFlow.

Sequence-to-sequence recurrent neural networks can learn complex functions
that map input sequences to output sequences. These models yield very good
results on a number of tasks, such as speech recognition, parsing, machine
translation, or even constructing automated replies to emails.

Before using this module, it is recommended to read the TensorFlow tutorial
on sequence-to-sequence models. It explains the basic concepts of this module
and shows an end-to-end example of how to build a translation model.
  https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html

Here is an overview of functions available in this module. They all use
a very similar interface, so after reading the above tutorial and using
one of them, others should be easy to substitute.

* Full sequence-to-sequence models.
  - basic_rnn_seq2seq: The most basic RNN-RNN model.
  - tied_rnn_seq2seq: The basic model with tied encoder and decoder weights.
  - embedding_rnn_seq2seq: The basic model with input embedding.
  - embedding_tied_rnn_seq2seq: The tied model with input embedding.
  - embedding_attention_seq2seq: Advanced model with input embedding and
      the neural attention mechanism; recommended for complex tasks.

* Multi-task sequence-to-sequence models.
  - one2many_rnn_seq2seq: The embedding model with multiple decoders.

* Decoders (when you write your own encoder, you can use these to decode;
    e.g., if you want to write a model that generates captions for images).
  - rnn_decoder: The basic decoder based on a pure RNN.
  - attention_decoder: A decoder that uses the attention mechanism.

* Losses.
  - sequence_loss: Loss for a sequence model returning average log-perplexity.
  - sequence_loss_by_example: As above, but not averaging over all examples.

* model_with_buckets: A convenience function to create models with bucketing
    (see the tutorial above for an explanation of why and how to use it).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = rnn_cell_impl._linear  # pylint: disable=protected-access


def _extract_argmax_and_embed(embedding,
                              output_projection=None,
                              update_embedding=True):
  """Get a loop_function that extracts the previous symbol and embeds it.

  Args:
    embedding: embedding tensor for symbols.
    output_projection: None or a pair (W, B). If provided, each fed previous
      output will first be multiplied by W and added B.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev, _):
    if output_projection is not None:
      prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = array_ops.stop_gradient(emb_prev)
    return emb_prev

  return loop_function


def rnn_decoder(decoder_inputs,
                initial_state,
                cell,
                loop_function=None,
                scope=None):
  """RNN decoder for the sequence-to-sequence model.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor with shape [batch_size x cell.state_size].
    cell: rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to the i-th output
      in order to generate the i+1-st input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing generated outputs.
      state: The state of each cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
        (Note that in some cases, like basic RNN cell or GRU cell, outputs and
         states can be the same. They are different for LSTM cells though.)
  """
  with variable_scope.variable_scope(scope or "rnn_decoder"):
    state = initial_state
    outputs = []
    prev = None
    for i, inp in enumerate(decoder_inputs):
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      output, state = cell(inp, state)
      outputs.append(output)
      if loop_function is not None:
        prev = output
  return outputs, state


def basic_rnn_seq2seq(encoder_inputs,
                      decoder_inputs,
                      cell,
                      dtype=dtypes.float32,
                      scope=None):
  """Basic RNN sequence-to-sequence model.

  This model first runs an RNN to encode encoder_inputs into a state vector,
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell type, but don't share parameters.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size].
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    dtype: The dtype of the initial state of the RNN cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "basic_rnn_seq2seq".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(scope or "basic_rnn_seq2seq"):
    enc_cell = copy.deepcopy(cell)
    _, enc_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
    return rnn_decoder(decoder_inputs, enc_state, cell)


def tied_rnn_seq2seq(encoder_inputs,
                     decoder_inputs,
                     cell,
                     loop_function=None,
                     dtype=dtypes.float32,
                     scope=None):
  """RNN sequence-to-sequence model with tied encoder and decoder parameters.

  This model first runs an RNN to encode encoder_inputs into a state vector, and
  then runs decoder, initialized with the last encoder state, on decoder_inputs.
  Encoder and decoder use the same RNN cell and share parameters.

  Args:
    encoder_inputs: A list of 2D Tensors [batch_size x input_size].
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol), see rnn_decoder for details.
    dtype: The dtype of the initial state of the rnn cell (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "tied_rnn_seq2seq".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope("combined_tied_rnn_seq2seq"):
    scope = scope or "tied_rnn_seq2seq"
    _, enc_state = rnn.static_rnn(
        cell, encoder_inputs, dtype=dtype, scope=scope)
    variable_scope.get_variable_scope().reuse_variables()
    return rnn_decoder(
        decoder_inputs,
        enc_state,
        cell,
        loop_function=loop_function,
        scope=scope)


def embedding_rnn_decoder(decoder_inputs,
                          initial_state,
                          cell,
                          num_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          update_embedding_for_previous=True,
                          scope=None):
  """RNN decoder with embedding and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each fed
      previous output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_decoder".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_decoder") as scope:
    if output_projection is not None:
      dtype = scope.dtype
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = (embedding_ops.embedding_lookup(embedding, i)
               for i in decoder_inputs)
    return rnn_decoder(
        emb_inp, initial_state, cell, loop_function=loop_function)


def embedding_rnn_seq2seq(encoder_inputs,
                          decoder_inputs,
                          cell,
                          num_encoder_symbols,
                          num_decoder_symbols,
                          embedding_size,
                          output_projection=None,
                          feed_previous=False,
                          dtype=None,
                          scope=None):
  """Embedding RNN sequence-to-sequence model.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_rnn_seq2seq"

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors. The
        output is of shape [batch_size x cell.output_size] when
        output_projection is not None (and represents the dense representation
        of predicted tokens). It is of shape [batch_size x num_decoder_symbols]
        when output_projection is None.
      state: The state of each decoder cell in each time-step. This is a list
        with length len(decoder_inputs) -- one item for each time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(scope or "embedding_rnn_seq2seq") as scope:
    if dtype is not None:
      scope.set_dtype(dtype)
    else:
      dtype = scope.dtype

    # Encoder.
    encoder_cell = copy.deepcopy(cell)
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.static_rnn(encoder_cell, encoder_inputs, dtype=dtype)

    # Decoder.
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)

    if isinstance(feed_previous, bool):
      return embedding_rnn_decoder(
          decoder_inputs,
          encoder_state,
          cell,
          num_decoder_symbols,
          embedding_size,
          output_projection=output_projection,
          feed_previous=feed_previous)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = embedding_rnn_decoder(
            decoder_inputs,
            encoder_state,
            cell,
            num_decoder_symbols,
            embedding_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state


def embedding_tied_rnn_seq2seq(encoder_inputs,
                               decoder_inputs,
                               cell,
                               num_symbols,
                               embedding_size,
                               num_decoder_symbols=None,
                               output_projection=None,
                               feed_previous=False,
                               dtype=None,
                               scope=None):
  """Embedding RNN sequence-to-sequence model with tied (shared) parameters.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_symbols x input_size]). Then it runs an RNN to encode embedded
  encoder_inputs into a state vector. Next, it embeds decoder_inputs using
  the same embedding. Then it runs RNN decoder, initialized with the last
  encoder state, on embedded decoder_inputs. The decoder output is over symbols
  from 0 to num_decoder_symbols - 1 if num_decoder_symbols is none; otherwise it
  is over 0 to num_symbols - 1.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_symbols: Integer; number of symbols for both encoder and decoder.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_decoder_symbols: Integer; number of output symbols for decoder. If
      provided, the decoder output is over symbols 0 to num_decoder_symbols - 1.
      Otherwise, decoder output is over symbols 0 to num_symbols - 1. Note that
      this assumes that the vocabulary is set up such that the first
      num_decoder_symbols of num_symbols are part of decoding.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has
      shape [num_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype to use for the initial RNN states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_tied_rnn_seq2seq".

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_symbols] containing the generated
        outputs where output_symbols = num_decoder_symbols if
        num_decoder_symbols is not None otherwise output_symbols = num_symbols.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  with variable_scope.variable_scope(
      scope or "embedding_tied_rnn_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype

    if output_projection is not None:
      proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
      proj_weights.get_shape().assert_is_compatible_with([None, num_symbols])
      proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
      proj_biases.get_shape().assert_is_compatible_with([num_symbols])

    embedding = variable_scope.get_variable(
        "embedding", [num_symbols, embedding_size], dtype=dtype)

    emb_encoder_inputs = [
        embedding_ops.embedding_lookup(embedding, x) for x in encoder_inputs
    ]
    emb_decoder_inputs = [
        embedding_ops.embedding_lookup(embedding, x) for x in decoder_inputs
    ]

    output_symbols = num_symbols
    if num_decoder_symbols is not None:
      output_symbols = num_decoder_symbols
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, output_symbols)

    if isinstance(feed_previous, bool):
      loop_function = _extract_argmax_and_embed(embedding, output_projection,
                                                True) if feed_previous else None
      return tied_rnn_seq2seq(
          emb_encoder_inputs,
          emb_decoder_inputs,
          cell,
          loop_function=loop_function,
          dtype=dtype)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      loop_function = _extract_argmax_and_embed(
          embedding, output_projection, False) if feed_previous_bool else None
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = tied_rnn_seq2seq(
            emb_encoder_inputs,
            emb_decoder_inputs,
            cell,
            loop_function=loop_function,
            dtype=dtype)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    # Calculate zero-state to know it's structure.
    static_batch_size = encoder_inputs[0].get_shape()[0]
    for inp in encoder_inputs[1:]:
      static_batch_size.merge_with(inp.get_shape()[0])
    batch_size = static_batch_size.value
    if batch_size is None:
      batch_size = array_ops.shape(encoder_inputs[0])[0]
    zero_state = cell.zero_state(batch_size, dtype)
    if nest.is_sequence(zero_state):
      state = nest.pack_sequence_as(
          structure=zero_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
  """RNN decoder with attention for the sequence-to-sequence model.

  In this context "attention" means that, during decoding, the RNN can look up
  information in the additional tensor attention_states, and it does this by
  focusing on a few entries from the tensor. This model has proven to yield
  especially good results in a number of sequence-to-sequence tasks. This
  implementation is based on http://arxiv.org/abs/1412.7449 (see below for
  details). It is recommended for complex sequence-to-sequence tasks.

  Args:
    decoder_inputs: A list of 2D Tensors [batch_size x input_size].
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    output_size: Size of the output vectors; if None, we use cell.output_size.
    num_heads: Number of attention heads that read from attention_states.
    loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and decoder_inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x input_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of decoder_inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if not decoder_inputs:
    raise ValueError("Must provide at least 1 input to attention decoder.")
  if num_heads < 1:
    raise ValueError("With less than 1 heads, use a non-attention decoder.")
  if attention_states.get_shape()[2].value is None:
    raise ValueError("Shape[2] of attention_states must be known: %s" %
                     attention_states.get_shape())
  if output_size is None:
    output_size = cell.output_size

  with variable_scope.variable_scope(
      scope or "attention_decoder", dtype=dtype) as scope:
    dtype = scope.dtype

    batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
    attn_length = attention_states.get_shape()[1].value
    if attn_length is None:
      attn_length = array_ops.shape(attention_states)[1]
    attn_size = attention_states.get_shape()[2].value

    # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
    hidden = array_ops.reshape(attention_states,
                               [-1, attn_length, 1, attn_size])
    hidden_features = []
    v = []
    attention_vec_size = attn_size  # Size of query vectors for attention.
    for a in xrange(num_heads):
      k = variable_scope.get_variable("AttnW_%d" % a,
                                      [1, 1, attn_size, attention_vec_size])
      hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
      v.append(
          variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

    state = initial_state

    def attention(query):
      """Put attention masks on hidden using hidden_features and query."""
      ds = []  # Results of attention reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = array_ops.concat(query_list, 1)
      for a in xrange(num_heads):
        with variable_scope.variable_scope("Attention_%d" % a):
          y = linear(query, attention_vec_size, True)
          y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
          # Attention mask is a softmax of v^T * tanh(...).
          s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                  [2, 3])
          a = nn_ops.softmax(s)
          # Now calculate the attention-weighted vector d.
          d = math_ops.reduce_sum(
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
          ds.append(array_ops.reshape(d, [-1, attn_size]))
      return ds

    outputs = []
    prev = None
    batch_attn_size = array_ops.stack([batch_size, attn_size])
    attns = [
        array_ops.zeros(
            batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
    ]
    for a in attns:  # Ensure the second shape of attention vectors is set.
      a.set_shape([None, attn_size])
    if initial_state_attention:
      attns = attention(initial_state)
    for i, inp in enumerate(decoder_inputs):
      if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
      # If loop_function is set, we use it instead of decoder_inputs.
      if loop_function is not None and prev is not None:
        with variable_scope.variable_scope("loop_function", reuse=True):
          inp = loop_function(prev, i)
      # Merge input and previous attentions into one vector of the right size.
      input_size = inp.get_shape().with_rank(2)[1]
      if input_size.value is None:
        raise ValueError("Could not infer input size from input: %s" % inp.name)
      x = linear([inp] + attns, input_size, True)
      # Run the RNN.
      cell_output, state = cell(x, state)
      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with variable_scope.variable_scope(
            variable_scope.get_variable_scope(), reuse=True):
          attns = attention(state)
      else:
        attns = attention(state)

      with variable_scope.variable_scope("AttnOutputProjection"):
        output = linear([cell_output] + attns, output_size, True)
      if loop_function is not None:
        prev = output
      outputs.append(output)

  return outputs, state


def embedding_attention_decoder(decoder_inputs,
                                initial_state,
                                attention_states,
                                cell,
                                num_symbols,
                                embedding_size,
                                num_heads=1,
                                output_size=None,
                                output_projection=None,
                                feed_previous=False,
                                update_embedding_for_previous=True,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """RNN decoder with embedding and attention and a pure-decoding option.

  Args:
    decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
    initial_state: 2D Tensor [batch_size x cell.state_size].
    attention_states: 3D Tensor [batch_size x attn_length x attn_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function.
    num_symbols: Integer, how many symbols come into the embedding.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_size: Size of the output vectors; if None, use output_size.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_symbols] and B has shape
      [num_symbols]; if provided and feed_previous=True, each fed previous
      output will first be multiplied by W and added B.
    feed_previous: Boolean; if True, only the first of decoder_inputs will be
      used (the "GO" symbol), and all other decoder inputs will be generated by:
        next = embedding_lookup(embedding, argmax(previous_output)),
      In effect, this implements a greedy decoder. It can also be used
      during training to emulate http://arxiv.org/abs/1506.03099.
      If False, decoder_inputs are used as given (the standard decoder case).
    update_embedding_for_previous: Boolean; if False and feed_previous=True,
      only the embedding for the first symbol of decoder_inputs (the "GO"
      symbol) will be updated by back propagation. Embeddings for the symbols
      generated from the decoder itself remain unchanged. This parameter has
      no effect if feed_previous=False.
    dtype: The dtype to use for the RNN initial states (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x output_size] containing the generated outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: When output_projection has the wrong shape.
  """
  if output_size is None:
    output_size = cell.output_size
  if output_projection is not None:
    proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
    proj_biases.get_shape().assert_is_compatible_with([num_symbols])

  with variable_scope.variable_scope(
      scope or "embedding_attention_decoder", dtype=dtype) as scope:

    embedding = variable_scope.get_variable("embedding",
                                            [num_symbols, embedding_size])
    loop_function = _extract_argmax_and_embed(
        embedding, output_projection,
        update_embedding_for_previous) if feed_previous else None
    emb_inp = [
        embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
    ]
    return attention_decoder(
        emb_inp,
        initial_state,
        attention_states,
        cell,
        output_size=output_size,
        num_heads=num_heads,
        loop_function=loop_function,
        initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                encoder_cell,
                                cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Warning: when output_projection is None, the size of the attention vectors
  and variables will be made proportional to num_decoder_symbols, can be large.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(
      scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.
    #encoder_cell = copy.deepcopy(cell)
    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    encoder_outputs, encoder_state = rnn.static_rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [
        array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
    ]
    attention_states = array_ops.concat(top_states, 1)

    # Decoder.
    output_size = None
    if output_projection is None:
      cell = core_rnn_cell.OutputProjectionWrapper(cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          decoder_inputs,
          encoder_state,
          attention_states,
          cell,
          num_decoder_symbols,
          embedding_size,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state


def one2many_rnn_seq2seq(encoder_inputs,
                         decoder_inputs_dict,
                         enc_cell,
                         dec_cells_dict,
                         num_encoder_symbols,
                         num_decoder_symbols_dict,
                         embedding_size,
                         feed_previous=False,
                         dtype=None,
                         scope=None):
  """One-to-many RNN sequence-to-sequence model (multi-task).

  This is a multi-task sequence-to-sequence model with one encoder and multiple
  decoders. Reference to multi-task sequence-to-sequence learning can be found
  here: http://arxiv.org/abs/1511.06114

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs_dict: A dictionary mapping decoder name (string) to
      the corresponding decoder_inputs; each decoder_inputs is a list of 1D
      Tensors of shape [batch_size]; num_decoders is defined as
      len(decoder_inputs_dict).
    enc_cell: tf.nn.rnn_cell.RNNCell defining the encoder cell function and
      size.
    dec_cells_dict: A dictionary mapping encoder name (string) to an
      instance of tf.nn.rnn_cell.RNNCell.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols_dict: A dictionary mapping decoder name (string) to an
      integer specifying number of symbols for the corresponding decoder;
      len(num_decoder_symbols_dict) must be equal to num_decoders.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first of
      decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial state for both the encoder and encoder
      rnn cells (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "one2many_rnn_seq2seq"

  Returns:
    A tuple of the form (outputs_dict, state_dict), where:
      outputs_dict: A mapping from decoder name (string) to a list of the same
        length as decoder_inputs_dict[name]; each element in the list is a 2D
        Tensors with shape [batch_size x num_decoder_symbol_list[name]]
        containing the generated outputs.
      state_dict: A mapping from decoder name (string) to the final state of the
        corresponding decoder RNN; it is a 2D Tensor of shape
        [batch_size x cell.state_size].

  Raises:
    TypeError: if enc_cell or any of the dec_cells are not instances of RNNCell.
    ValueError: if len(dec_cells) != len(decoder_inputs_dict).
  """
  outputs_dict = {}
  state_dict = {}

  if not isinstance(enc_cell, rnn_cell_impl.RNNCell):
    raise TypeError("enc_cell is not an RNNCell: %s" % type(enc_cell))
  if set(dec_cells_dict) != set(decoder_inputs_dict):
    raise ValueError("keys of dec_cells_dict != keys of decodre_inputs_dict")
  for dec_cell in dec_cells_dict.values():
    if not isinstance(dec_cell, rnn_cell_impl.RNNCell):
      raise TypeError("dec_cell is not an RNNCell: %s" % type(dec_cell))

  with variable_scope.variable_scope(
      scope or "one2many_rnn_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype

    # Encoder.
    enc_cell = core_rnn_cell.EmbeddingWrapper(
        enc_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    _, encoder_state = rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)

    # Decoder.
    for name, decoder_inputs in decoder_inputs_dict.items():
      num_decoder_symbols = num_decoder_symbols_dict[name]
      dec_cell = dec_cells_dict[name]

      with variable_scope.variable_scope("one2many_decoder_" + str(
          name)) as scope:
        dec_cell = core_rnn_cell.OutputProjectionWrapper(
            dec_cell, num_decoder_symbols)
        if isinstance(feed_previous, bool):
          outputs, state = embedding_rnn_decoder(
              decoder_inputs,
              encoder_state,
              dec_cell,
              num_decoder_symbols,
              embedding_size,
              feed_previous=feed_previous)
        else:
          # If feed_previous is a Tensor, we construct 2 graphs and use cond.
          def filled_embedding_rnn_decoder(feed_previous):
            """The current decoder with a fixed feed_previous parameter."""
            # pylint: disable=cell-var-from-loop
            reuse = None if feed_previous else True
            vs = variable_scope.get_variable_scope()
            with variable_scope.variable_scope(vs, reuse=reuse):
              outputs, state = embedding_rnn_decoder(
                  decoder_inputs,
                  encoder_state,
                  dec_cell,
                  num_decoder_symbols,
                  embedding_size,
                  feed_previous=feed_previous)
            # pylint: enable=cell-var-from-loop
            state_list = [state]
            if nest.is_sequence(state):
              state_list = nest.flatten(state)
            return outputs + state_list

          outputs_and_state = control_flow_ops.cond(
              feed_previous, lambda: filled_embedding_rnn_decoder(True),
              lambda: filled_embedding_rnn_decoder(False))
          # Outputs length is the same as for decoder inputs.
          outputs_len = len(decoder_inputs)
          outputs = outputs_and_state[:outputs_len]
          state_list = outputs_and_state[outputs_len:]
          state = state_list[0]
          if nest.is_sequence(encoder_state):
            state = nest.pack_sequence_as(
                structure=encoder_state, flat_sequence=state_list)
      outputs_dict[name] = outputs
      state_dict[name] = state

  return outputs_dict, state_dict


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None,
                             name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.name_scope(name, "sequence_loss_by_example",
                      logits + targets + weights):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            labels=target, logits=logit)
      else:
        crossent = softmax_loss_function(labels=target, logits=logit)
      log_perp_list.append(crossent * weight)
    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  with ops.name_scope(name, "sequence_loss", logits + targets + weights):
    cost = math_ops.reduce_sum(
        sequence_loss_by_example(
            logits,
            targets,
            weights,
            average_across_timesteps=average_across_timesteps,
            softmax_loss_function=softmax_loss_function))
    if average_across_batch:
      batch_size = array_ops.shape(targets[0])[0]
      return cost / math_ops.cast(batch_size, cost.dtype)
    else:
      return cost


def model_with_buckets(encoder_inputs,
                       decoder_inputs,
                       targets,
                       weights,
                       buckets,
                       seq2seq,
                       softmax_loss_function=None,
                       per_example_loss=False,
                       name=None):
  """Create a sequence-to-sequence model with support for bucketing.

  The seq2seq argument is a function that defines a sequence-to-sequence model,
  e.g., seq2seq = lambda x, y: basic_rnn_seq2seq(
      x, y, rnn_cell.GRUCell(24))

  Args:
    encoder_inputs: A list of Tensors to feed the encoder; first seq2seq input.
    decoder_inputs: A list of Tensors to feed the decoder; second seq2seq input.
    targets: A list of 1D batch-sized int32 Tensors (desired output sequence).
    weights: List of 1D batch-sized float-Tensors to weight the targets.
    buckets: A list of pairs of (input size, output size) for each bucket.
    seq2seq: A sequence-to-sequence model function; it takes 2 input that
      agree with encoder_inputs and decoder_inputs, and returns a pair
      consisting of outputs and states (as, e.g., basic_rnn_seq2seq).
    softmax_loss_function: Function (labels, logits) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
      **Note that to avoid confusion, it is required for the function to accept
      named arguments.**
    per_example_loss: Boolean. If set, the returned loss will be a batch-sized
      tensor of losses for each sequence in the batch. If unset, it will be
      a scalar with the averaged loss from all examples.
    name: Optional name for this operation, defaults to "model_with_buckets".

  Returns:
    A tuple of the form (outputs, losses), where:
      outputs: The outputs for each bucket. Its j'th element consists of a list
        of 2D Tensors. The shape of output tensors can be either
        [batch_size x output_size] or [batch_size x num_decoder_symbols]
        depending on the seq2seq model used.
      losses: List of scalar Tensors, representing losses for each bucket, or,
        if per_example_loss is set, a list of 1D batch-sized float Tensors.

  Raises:
    ValueError: If length of encoder_inputs, targets, or weights is smaller
      than the largest (last) bucket.
  """
  if len(encoder_inputs) < buckets[-1][0]:
    raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                     "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
  if len(targets) < buckets[-1][1]:
    raise ValueError("Length of targets (%d) must be at least that of last "
                     "bucket (%d)." % (len(targets), buckets[-1][1]))
  if len(weights) < buckets[-1][1]:
    raise ValueError("Length of weights (%d) must be at least that of last "
                     "bucket (%d)." % (len(weights), buckets[-1][1]))

  all_inputs = encoder_inputs + decoder_inputs + targets + weights
  losses = []
  outputs = []
  with ops.name_scope(name, "model_with_buckets", all_inputs):
    for j, bucket in enumerate(buckets):
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=True if j > 0 else None):
        bucket_outputs, _ = seq2seq(encoder_inputs[:bucket[0]],
                                    decoder_inputs[:bucket[1]])
        outputs.append(bucket_outputs)
        if per_example_loss:
          losses.append(
              sequence_loss_by_example(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))
        else:
          losses.append(
              sequence_loss(
                  outputs[-1],
                  targets[:bucket[1]],
                  weights[:bucket[1]],
                  softmax_loss_function=softmax_loss_function))

  return outputs, losses

```
#### seq2seq_model示例代码：
现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 seq2seq_model.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 seq2seq_model.py" />

```python
/// <example verb="create" file="/home/ubuntu/seq2seq_model.py" />
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange
import tensorflow as tf
import seq2seq
import generate_chat 


class Seq2SeqModel(object):

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               dtype=tf.float32):
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    output_projection = None
    softmax_loss_function = None
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=local_w_t,
                biases=local_b,
                labels=labels,
                inputs=local_inputs,
                num_sampled=num_samples,
                num_classes=self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss

    def single_cell():
      return tf.contrib.rnn.GRUCell(size)
    if use_lstm:
      def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(size)
    cell = single_cell()
    encoder_cell = single_cell()
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
      encoder_cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])

    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          encoder_cell,
          cell,
          num_encoder_symbols=source_vocab_size,
          num_decoder_symbols=target_vocab_size,
          embedding_size=size,
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=dtype)

    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="encoder{0}".format(i)))
    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))

    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    if forward_only:
      self.outputs, self.losses = seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data, bucket_id):
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [generate_chat.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([generate_chat.GO_ID] + decoder_input +
                            [generate_chat.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == generate_chat.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights


```

> <checker type="output-contains" command="ls /home/ubuntu/seq2seq.py" hint="请创建并执行 /home/ubuntu/seq2seq.py">
>    <keyword regex="/home/ubuntu/seq2seq.py" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/seq2seq_model.py" hint="请创建并执行 /home/ubuntu/seq2seq_model.py">
>    <keyword regex="/home/ubuntu/seq2seq_model.py" />
> </checker>

### 训练 Seq2Seq 模型
训练 30 万次后，损失函数基本保持不变，单个GPU 大概需要 17 个小时左右，如果采用 CPU 训练，大概需要 3 天左右。你可以调整循环次数，体验下训练过程，可以直接下载我们训练好的模型。

#### 示例代码：
现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 train_chat.py][create-example]，内容可参考：
> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 train_chat.py" />

```python
/// <example verb="create" file="/home/ubuntu/train_chat.py" />
#-*- coding:utf-8 -*-
import generate_chat
import seq2seq_model
import tensorflow as tf
import numpy as np
import logging
import logging.handlers

if __name__ == '__main__':

    _,_,source_vocab_size = generate_chat.get_vocabs(generate_chat.vocab_encode_file)
    _,_,target_vocab_size = generate_chat.get_vocabs(generate_chat.vocab_decode_file)
    train_set = generate_chat.read_data(generate_chat.train_encode_vec_file,generate_chat.train_decode_vec_file)
    test_set = generate_chat.read_data(generate_chat.test_encode_vec_file,generate_chat.test_decode_vec_file)
    train_bucket_sizes = [len(train_set[i]) for i in range(len(generate_chat._buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in range(len(train_bucket_sizes))]
    with tf.Session() as sess:
        model = seq2seq_model.Seq2SeqModel(source_vocab_size,
            target_vocab_size,
            generate_chat._buckets,
            generate_chat.units_num,
            generate_chat.num_layers,
            generate_chat.max_gradient_norm,
            generate_chat.batch_size,
            generate_chat.learning_rate,
            generate_chat.learning_rate_decay_factor,
            use_lstm = True)
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        loss = 0.0
        step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale)) if train_buckets_scale[i] > random_number_01])
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, False)
			print("step:%d,loss:%f" % (step,step_loss))
            loss += step_loss / 2000
            step += 1
            if step % 1000 == 0:
                print("step:%d,per_loss:%f" % (step,loss))
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                model.saver.save(sess, "chatbot.ckpt", global_step=model.global_step)
                loss = 0.0
            if step % 5000 == 0:
                for bucket_id in range(len(generate_chat._buckets)):
                    if len(test_set[bucket_id]) == 0:
                        continue
                        encoder_inputs, decoder_inputs, target_weights = model.get_batch(test_set, bucket_id)
                        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                        print("bucket_id:%d,eval_loss:%f" % (bucket_id,eval_loss))

```
> <checker type="output-contains" command="ls /home/ubuntu/train_chat.py" hint="请创建并执行 /home/ubuntu/train_chat.py">
>    <keyword regex="/home/ubuntu/train_chat.py" />
> </checker>

**然后执行:**

```
cd /home/ubuntu;
python train_chat.py
```
**执行结果：**
```
step:311991,loss:0.000332
step:311992,loss:0.000199
step:311993,loss:0.000600
step:311994,loss:0.001900
step:311995,loss:0.018695
step:311996,loss:0.000945
step:311997,loss:0.000517
step:311998,loss:0.000530
step:311999,loss:0.001020
step:312000,per_loss:0.000672
step:312000,loss:0.000276
step:312001,loss:0.000332
step:312002,loss:0.003255
step:312003,loss:0.000452
step:312004,loss:0.000553
```

**下载已有模型:**
```
wget http://tensorflow-1253902462.cosgz.myqcloud.com/rnn_chat/chat_model.zip
unzip -o chat_model.zip
```
> <checker type="output-contains" command="ls /home/ubuntu/chat_model.zip" hint="下载 chat_model.zip">
>    <keyword regex="/home/ubuntu/chat_model.zip" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/chatbot.ckpt-317000.data-00000-of-00001" hint="请解压 chat_model.zip">
>    <keyword regex="/home/ubuntu/chatbot.ckpt-317000.data-00000-of-00001" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/chatbot.ckpt-317000.index" hint="请解压 chat_model.zip">
>    <keyword regex="/home/ubuntu/chatbot.ckpt-317000.index" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/vocab_decode" hint="请解压 chat_model.zip">
>    <keyword regex="/home/ubuntu/vocab_decode" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/vocab_encode" hint="请解压 chat_model.zip">
>    <keyword regex="/home/ubuntu/vocab_encode" />
> </checker>


### 开始聊天
利用训练好的模型，我们可以开始聊天了。训练数据有限只能进行简单的对话，提问最好参考训练数据，否则效果不理想。

**示例代码：**

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 predict_chat.py][create-example]，内容可参考：
> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 predict_chat.py" />

```python
/// <example verb="create" file="/home/ubuntu/predict_chat.py" />
#-*- coding:utf-8 -*-
import generate_chat
import seq2seq_model
import tensorflow as tf
import numpy as np
import sys

if __name__ == '__main__':
    source_id_to_word,source_word_to_id,source_vocab_size = generate_chat.get_vocabs(generate_chat.vocab_encode_file)
    target_id_to_word,target_word_to_id,target_vocab_size = generate_chat.get_vocabs(generate_chat.vocab_decode_file)
    to_id = lambda word: source_word_to_id.get(word,generate_chat.UNK_ID)
    with tf.Session() as sess:
        model = seq2seq_model.Seq2SeqModel(source_vocab_size,
                                           target_vocab_size,
                                           generate_chat._buckets,
                                           generate_chat.units_num,
                                           generate_chat.num_layers,
                                           generate_chat.max_gradient_norm,
                                           1,
                                           generate_chat.learning_rate,
                                           generate_chat.learning_rate_decay_factor,
                                           forward_only = True,
                                           use_lstm = True)
        model.saver.restore(sess,"chatbot.ckpt-317000")
        while True:
            sys.stdout.write("ask > ")
            sys.stdout.flush()
            sentence = sys.stdin.readline().strip('\\n')
            flag = generate_chat.is_chinese(sentence)
            if not sentence or not flag:
              print("请输入纯中文")
              continue
            sentence_vec = list(map(to_id,sentence))
            bucket_id = len(generate_chat._buckets) - 1
            if len(sentence_vec) > generate_chat._buckets[bucket_id][0]:
                print("sentence too long max:%d" % generate_chat._buckets[bucket_id][0])
                exit(0)
            for i,bucket in enumerate(generate_chat._buckets):
                if bucket[0] >= len(sentence_vec):
                    bucket_id = i
                    break
            encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(sentence_vec, [])]}, bucket_id)
            _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
            outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
            if generate_chat.EOS_ID in outputs:
                outputs = outputs[:outputs.index(generate_chat.EOS_ID)]
            answer = "".join([tf.compat.as_str(target_id_to_word[output]) for output in outputs])
            print("answer > " + answer)

```
> <checker type="output-contains" command="ls /home/ubuntu/predict_chat.py" hint="请创建并执行 /home/ubuntu/predict_chat.py">
>    <keyword regex="/home/ubuntu/predict_chat.py" />
> </checker>

**然后执行（需要耐心等待几分钟）:**

```
cd /home/ubuntu
python predict_chat.py
```
**执行结果：**
```
ask > 你大爷
answer > 你大爷
ask > 你好
answer > 你好呀
ask > 我是谁
answer > 哈哈，大屌丝，地地眼
```
## 完成实验

### 实验内容已完成

您可进行更多关于机器学习教程：

* [实验列表 - 机器学习][https://cloud.tencent.com/developer/labs/gallery?tagId=31]

关于 TensorFlow 的更多资料可参考 [TensorFlow 官网 ][https://www.tensorflow.org/]。