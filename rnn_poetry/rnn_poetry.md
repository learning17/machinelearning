TensorFlow - 基于 RNN 生成古诗词
============================

## 简介

基于 TensoFlow 构建两层的 RNN，采用 4 万多首唐诗作为训练数据，实现可以写古诗的 AI demo。

### 步骤简介

本教程一共分为四个部分

* `generate_poetry.py` - 古诗清洗、过滤较长或较短古诗、过滤即非五言也非七言古诗、为每个字生成唯一的数字ID、每首古诗用数字ID表示；
* `poetry_model.py` - 两层RNN网络模型，采用LSTM模型；
* `train_poetry.py` - 训练LSTM模型；
* `predict_poetry.py` - 生成古诗，随机取一个汉字，根据该汉字生成一首古诗。

## 数据学习

### 获取训练数据

我们在腾讯云的 COS 上准备了 4 万首古诗数据，使用 `wget` 命令获取：

```
wget http://tensorflow-1253902462.cosgz.myqcloud.com/rnn_poetry/poetry
```

> <checker type="output-contains" command="ls /home/ubuntu/poetry" hint="请使用 wget 下载古诗数据">
>   <keyword regex="/home/ubuntu/poetry" />
> </checker>

### 数据预处理

#### 处理思路：

* 数据中的每首唐诗以 `\[` 开头、`\]` 结尾，后续生成古诗时，根据 `\[` 随机取一个字，根据 `\]` 判断是否结束。
* 两种词袋：“汉字 => 数字”、“数字 => 汉字”，根据第一个词袋将每首古诗转化为数字表示。
* 诗歌的生成是根据上一个汉字生成下一个汉字，所以 `x_batch` 和 `y_batch` 的 `shape` 是相同的，`y_batch` 是 `x_batch` 中每一位向前循环移动一位。前面介绍每首唐诗 `\[`开头、`\]` 结尾，在这里也体现出好处，`\]` 下一个一定是 `\[`（即一首诗结束下一首诗开始）

具体可以看下面例子：

```
x_batch：['[', 12, 23, 34, 45, 56, 67, 78, ']']
y_batch：[12, 23, 34, 45, 56, 67, 78, ']', '[']
```

#### 示例代码：

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 generate_poetry.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 generate_poetry.py" />

```python
/// <example verb="create" file="/home/ubuntu/generate_poetry.py" />
#-*- coding:utf-8 -*-
import numpy as np
from io import open
import sys
import collections
reload(sys)
sys.setdefaultencoding('utf8')

class Poetry:
    def __init__(self):
        self.filename = "poetry"
        self.poetrys = self.get_poetrys()
        self.poetry_vectors,self.word_to_id,self.id_to_word = self.gen_poetry_vectors()
        self.poetry_vectors_size = len(self.poetry_vectors)
        self._index_in_epoch = 0

    def get_poetrys(self):
        poetrys = list()
        f = open(self.filename,"r", encoding='utf-8')
        for line in f.readlines():
            _,content = line.strip('\\n').strip().split(':')
            content = content.replace(' ','')
            #过滤含有特殊符号的唐诗
            if(not content or '_' in content or '(' in content or '（' in content or "□" in content
                   or '《' in content or '[' in content or ':' in content or '：'in content):
                continue
            #过滤较长或较短的唐诗
            if len(content) < 5 or len(content) > 79:
                continue
            content_list = content.replace('，', '|').replace('。', '|').split('|')
            flag = True
            #过滤即非五言也非七验的唐诗
            for sentence in content_list:
                slen = len(sentence)
                if 0 == slen:
                    continue
                if 5 != slen and 7 != slen:
                    flag = False
                    break
            if flag:
                #每首古诗以'['开头、']'结尾
                poetrys.append('[' + content + ']')
        return poetrys

    def gen_poetry_vectors(self):
        words = sorted(set(''.join(self.poetrys) + ' '))
        #数字ID到每个字的映射
        id_to_word = {i: word for i, word in enumerate(words)}
        #每个字到数字ID的映射
        word_to_id = {v: k for k, v in id_to_word.items()}
        to_id = lambda word: word_to_id.get(word)
        #唐诗向量化
        poetry_vectors = [list(map(to_id, poetry)) for poetry in self.poetrys]
        return poetry_vectors,word_to_id,id_to_word

    def next_batch(self,batch_size):
        assert batch_size < self.poetry_vectors_size
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        #取完一轮数据，打乱唐诗集合，重新取数据
        if self._index_in_epoch > self.poetry_vectors_size:
            np.random.shuffle(self.poetry_vectors)
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        batches = self.poetry_vectors[start:end]
        x_batch = np.full((batch_size, max(map(len, batches))), self.word_to_id[' '], np.int32)
        for row in range(batch_size):
            x_batch[row,:len(batches[row])] = batches[row]
        y_batch = np.copy(x_batch)
        y_batch[:,:-1] = x_batch[:,1:]
        y_batch[:,-1] = x_batch[:, 0]

        return x_batch,y_batch
```

下面我们可以看下预处理后的数据长啥样，可以在终端中一步一步执行下面命令：

启动 python：

```
python
```

构建数据：

```python
from generate_poetry import Poetry
p = Poetry()
```

查看第一首唐诗数字表示（[查看输出][output-1]）：

```python
print(p.poetry_vectors[0])
```

> <bubble for="output-1">
> *输出*：[1, 1101, 5413, 3437, 1416, 555, 5932, 1965, 5029, 5798, 889, 1357, 3, 397, 5567, 5576, 1285, 2143, 5932, 1985, 5449, 5332, 4092, 2198, 3, 3314, 2102, 5483, 1940, 3475, 5932, 3750, 2467, 3863, 1913, 4110, 3, 4081, 3081, 397, 5432, 542, 5932, 3737, 2157, 1254, 4205, 2082, 3, 2]
> </bubble>

根据 ID 查看对应的汉字（[查看输出][output-2]）：

```python
print(p.id_to_word[1101])
```

> <bubble for="output-2">
> *输出*：寒
> </bubble>

根据汉字查看对应的数字（[查看输出][output-3]）：

```python
print(p.word_to_id[u"寒"])
```

> <bubble for="output-3">
> *输出*：1101
> </bubble>

查看 x_batch、y_batch（[查看输出][output-4]）：

```python
x_batch, y_batch = p.next_batch(1)
x_batch
y_batch
```

> <bubble for="output-4">
> x_batch [   1, 1101, 5413, 3437, 1416,  555, 5932, 1965, 5029, 5798,  889,
>   1357,    3,  397, 5567, 5576, 1285, 2143, 5932, 1985, 5449, 5332,
>   4092, 2198,    3, 3314, 2102, 5483, 1940, 3475, 5932, 3750, 2467,
>   3863, 1913, 4110,    3, 4081, 3081,  397, 5432,  542, 5932, 3737,
>   2157, 1254, 4205, 2082,    3,    2]
>
> y_batch [1101, 5413, 3437, 1416,  555, 5932, 1965, 5029, 5798,  889, 1357,
>      3,  397, 5567, 5576, 1285, 2143, 5932, 1985, 5449, 5332, 4092,
>   2198,    3, 3314, 2102, 5483, 1940, 3475, 5932, 3750, 2467, 3863,
>   1913, 4110,    3, 4081, 3081,  397, 5432,  542, 5932, 3737, 2157,
>   1254, 4205, 2082,    3,    2,    1]
> </bubble>

> <checker type="output-contains" command="ls /home/ubuntu/generate_poetry.py" hint="请创建并执行 /home/ubuntu/generate_poetry.py">
>    <keyword regex="/home/ubuntu/generate_poetry.py" />
> </checker>

### LSTM 模型

上面我们将每个字用一个数字表示，但在模型训练过程中，需要对每个字进行向量化，Embedding 的作用按照 inputs 顺序返回 embedding 中的对应行，类似：

```python
import numpy as np
embedding = np.random.random([100, 10]) 
inputs = np.array([7, 17, 27, 37])
print(embedding[inputs])
```

#### 示例代码：

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 poetry_model.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 poetry_model.py" />

```python
/// <example verb="create" file="/home/ubuntu/poetry_model.py" />
#-*- coding:utf-8 -*-
import tensorflow as tf

class poetryModel:
    #定义权重和偏置项
    def rnn_variable(self,rnn_size,words_size):
        with tf.variable_scope('variable'):
            w = tf.get_variable("w", [rnn_size, words_size])
            b = tf.get_variable("b", [words_size])
        return w,b

    #损失函数
    def loss_model(self,words_size,targets,logits):
        targets = tf.reshape(targets,[-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],words_size)
        loss = tf.reduce_mean(loss)
        return loss

    #优化算子
    def optimizer_model(self,loss,learning_rate):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    #每个字向量化
    def embedding_variable(self,inputs,rnn_size,words_size):
        with tf.variable_scope('embedding'):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [words_size, rnn_size])
                input_data = tf.nn.embedding_lookup(embedding,inputs)
        return input_data
        
    #构建LSTM模型
    def create_model(self,inputs,batch_size,rnn_size,words_size,num_layers,is_training,keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_size,state_is_tuple=True)
        input_data = self.embedding_variable(inputs,rnn_size,words_size)
        if is_training:
            lstm = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            input_data = tf.nn.dropout(input_data,keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers,state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs,last_state = tf.nn.dynamic_rnn(cell,input_data,initial_state=initial_state)
        outputs = tf.reshape(outputs,[-1, rnn_size])
        w,b = self.rnn_variable(rnn_size,words_size)
        logits = tf.matmul(outputs,w) + b
        probs = tf.nn.softmax(logits)
        return logits,probs,initial_state,last_state
```
> <checker type="output-contains" command="ls /home/ubuntu/poetry_model.py" hint="请创建并执行 /home/ubuntu/poetry_model.py">
>    <keyword regex="/home/ubuntu/poetry_model.py" />
> </checker>

### 训练 LSTM 模型

每批次采用 50 首唐诗训练，训练 40000 次后，损失函数基本保持不变，GPU 大概需要 2 个小时左右。当然你可以调整循环次数，节省训练时间，亦或者直接下载我们训练好的模型。

```
wget http://tensorflow-1253902462.cosgz.myqcloud.com/rnn_poetry/poetry_model.zip
unzip poetry_model.zip
```

**示例代码：**

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 train_poetry.py][create-example]，内容可参考：
> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 train_poetry.py" />

```python
/// <example verb="create" file="/home/ubuntu/train_poetry.py" />
#-*- coding:utf-8 -*-
from generate_poetry import Poetry
from poetry_model import poetryModel
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    batch_size = 50
    epoch = 20
    rnn_size = 128
    num_layers = 2
    poetrys = Poetry()
    words_size = len(poetrys.word_to_id)
    inputs = tf.placeholder(tf.int32, [batch_size, None])
    targets = tf.placeholder(tf.int32, [batch_size, None])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    model = poetryModel()
    logits,probs,initial_state,last_state = model.create_model(inputs,batch_size,
                                                               rnn_size,words_size,num_layers,True,keep_prob)
    loss = model.loss_model(words_size,targets,logits)
    learning_rate = tf.Variable(0.0, trainable=False)
    optimizer = model.optimizer_model(loss,learning_rate)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(learning_rate, 0.002 * 0.97 ))
        next_state = sess.run(initial_state)
        step = 0
        while True:
            x_batch,y_batch = poetrys.next_batch(batch_size)
            feed = {inputs:x_batch,targets:y_batch,initial_state:next_state,keep_prob:0.5}
            train_loss, _ ,next_state = sess.run([loss,optimizer,last_state], feed_dict=feed)
            print("step:%d loss:%f" % (step,train_loss))
            if step > 40000:
                break
            if step%1000 == 0:
                n = step/1000
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** n)))
            step += 1
        saver.save(sess,"poetry_model.ckpt")
```

**然后执行（如果已下载模型，可以省略此步骤，不过建议自己修改循环次数体验下）:**

```
cd /home/ubuntu;
python train_poetry.py
```

**执行结果：**
```
step:0 loss:8.692488
step:1 loss:8.685234
step:2 loss:8.674787
step:3 loss:8.642109
step:4 loss:8.533745
step:5 loss:8.155352
step:6 loss:7.797368
step:7 loss:7.635432
step:8 loss:7.254006
step:9 loss:7.075273
step:10 loss:6.606557
step:11 loss:6.284406
step:12 loss:6.197527
step:13 loss:6.022724
step:14 loss:5.539262
step:15 loss:5.285880
step:16 loss:4.625040
step:17 loss:5.167739
```

> <checker type="output-contains" command="ls /home/ubuntu/poetry_model.ckpt.index" hint="下载模型或者自己训练">
>    <keyword regex="/home/ubuntu/poetry_model.ckpt.index" />
> </checker>

## 生成古诗

### 生成古诗

根据 `\[` 随机取一个汉字，作为生成古诗的第一个字，遇到 `\]` 结束生成古诗。

**示例代码：**

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 predict_poetry.py][create-example]，内容可参考：
> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 predict_poetry.py" />

```
/// <example verb="create" file="/home/ubuntu/predict_poetry.py" />
#-*- coding:utf-8 -*-
from generate_poetry import Poetry
from poetry_model import poetryModel
from operator import itemgetter
import tensorflow as tf
import numpy as np
import random


if __name__ == '__main__':
    batch_size = 1
    rnn_size = 128
    num_layers = 2
    poetrys = Poetry()
    words_size = len(poetrys.word_to_id)

    def to_word(prob):
        prob = prob[0]
        indexs, _ = zip(*sorted(enumerate(prob), key=itemgetter(1)))
        rand_num = int(np.random.rand(1)*10);
        index_sum = len(indexs)
        max_rate = prob[indexs[(index_sum-1)]]
        if max_rate > 0.9 :
            sample = indexs[(index_sum-1)]
        else:
            sample = indexs[(index_sum-1-rand_num)]
        return poetrys.id_to_word[sample]

    inputs = tf.placeholder(tf.int32, [batch_size, None])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    model = poetryModel()
    logits,probs,initial_state,last_state = model.create_model(inputs,batch_size,
                                                               rnn_size,words_size,num_layers,False,keep_prob)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,"poetry_model.ckpt")
        next_state = sess.run(initial_state)

        x = np.zeros((1, 1))
        x[0,0] = poetrys.word_to_id['[']
        feed = {inputs: x, initial_state: next_state, keep_prob: 1}
        predict, next_state = sess.run([probs, last_state], feed_dict=feed)
        word = to_word(predict)
        poem = ''
        while word != ']':
            poem += word
            x = np.zeros((1, 1))
            x[0, 0] = poetrys.word_to_id[word]
            feed = {inputs: x, initial_state: next_state, keep_prob: 1}
            predict, next_state = sess.run([probs, last_state], feed_dict=feed)
            word = to_word(predict)
        print poem
```
**然后执行:**

```
cd /home/ubuntu;
python predict_poetry.py
```

**执行结果：**

```
山风万仞下，寒雪入云空。风雪千家树，天花日晚深。秋来秋夜尽，风断雪山寒。莫道人无处，归人又可伤。
```
每次执行生成的古诗不一样，您可以多执行几次，看下实验结果。

> <checker type="output-contains" command="ls /home/ubuntu/predict_poetry.py" hint="请创建并执行 /home/ubuntu/predict_poetry.py">
>    <keyword regex="/home/ubuntu/predict_poetry.py" />
> </checker>

## 完成实验

### 实验内容已完成

您可进行更多关于机器学习教程：

* [实验列表 - 机器学习][https://cloud.tencent.com/developer/labs/gallery?tagId=31]

关于 TensorFlow 的更多资料可参考 [TensorFlow 官网 ][https://www.tensorflow.org/]。