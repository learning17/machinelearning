TensorFlow - 基于 CNN 破解验证码
============================

## 简介

传统的验证码识别算法一般需要把验证码分割为单个字符，然后逐个识别。本教程将验证码识别问题转化为分类的问题，实现对验证码进行整体识别。

### 步骤简介
本教程一共分为四个部分
* `generate_captcha.py` - 利用 Captcha 库生成验证码；
* `captcha_model.py` - CNN 模型；
* `train_captcha.py` - 训练 CNN 模型；
* `predict_captcha.py` - 识别验证码。

## 数据学习

### 获取训练数据

本教程使用的验证码由数字、大写字母、小写字母组成，每个验证码包含 4 个字符，总共有 62^4 种组合，所以一共有 62^4 种不同的验证码。

**示例代码：**

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 generate_captcha.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 generate_captcha.py" />

```python
/// <example verb="create" file="/home/ubuntu/generate_captcha.py" />
#-*- coding:utf-8 -*-
from captcha.image import ImageCaptcha
from PIL import Image
import numpy as np
import random
import string

class generateCaptcha():
    def __init__(self,
                 width = 160,#验证码图片的宽
                 height = 60,#验证码图片的高
                 char_num = 4,#验证码字符个数
                 characters = string.digits + string.ascii_uppercase + string.ascii_lowercase):#验证码组成，数字+大写字母+小写字母
        self.width = width
        self.height = height
        self.char_num = char_num
        self.characters = characters
        self.classes = len(characters)

    def gen_captcha(self,batch_size = 50):
        X = np.zeros([batch_size,self.height,self.width,1])
        img = np.zeros((self.height,self.width),dtype=np.uint8)
        Y = np.zeros([batch_size,self.char_num,self.classes])
        image = ImageCaptcha(width = self.width,height = self.height)

        while True:
            for i in range(batch_size):
                captcha_str = ''.join(random.sample(self.characters,self.char_num))
                img = image.generate_image(captcha_str).convert('L')
                img = np.array(img.getdata())
                X[i] = np.reshape(img,[self.height,self.width,1])/255.0
                for j,ch in enumerate(captcha_str):
                    Y[i,j,self.characters.find(ch)] = 1
            Y = np.reshape(Y,(batch_size,self.char_num*self.classes))
            yield X,Y

    def decode_captcha(self,y):
        y = np.reshape(y,(len(y),self.char_num,self.classes))
        return ''.join(self.characters[x] for x in np.argmax(y,axis = 2)[0,:])

    def get_parameter(self):
        return self.width,self.height,self.char_num,self.characters,self.classes

    def gen_test_captcha(self):
        image = ImageCaptcha(width = self.width,height = self.height)
        captcha_str = ''.join(random.sample(self.characters,self.char_num))
        img = image.generate_image(captcha_str)
        img.save(captcha_str + '.jpg')

        X = np.zeros([1,self.height,self.width,1])
        Y = np.zeros([1,self.char_num,self.classes])
        img = img.convert('L')
        img = np.array(img.getdata())
        X[0] = np.reshape(img,[self.height,self.width,1])/255.0
        for j,ch in enumerate(captcha_str):
            Y[0,j,self.characters.find(ch)] = 1
        Y = np.reshape(Y,(1,self.char_num*self.classes))
        return X,Y
```

> <checker type="output-contains" command="ls /home/ubuntu/generate_captcha.py" hint="请创建 /home/ubuntu/generate_captcha.py">
>    <keyword regex="/home/ubuntu/generate_captcha.py" />
> </checker>

### 理解训练数据

* X：一个 mini-batch 的训练数据，其 shape 为 [ batch_size, height, width, 1 ]，batch_size 表示每批次多少个训练数据，height 表示验证码图片的高，width 表示验证码图片的宽，1 表示图片的通道。
* Y：X 中每个训练数据属于哪一类验证码，其形状为 [ batch_size, class ] ，对验证码中每个字符进行 One-Hot 编码，所以 class 大小为 4*62。

**执行:**
* 获取验证码和对应的分类

```
cd /home/ubuntu;
python
from generate_captcha import generateCaptcha
g = generateCaptcha()
X,Y = g.gen_test_captcha()
```
* 查看训练数据

```
X.shape
Y.shape
```
可以在 [ /home/ubuntu ][locate_data] 目录下查看生成的验证码，jpg 格式的图片可以点击查看。
> <locate for="locate_data" path="/home/ubuntu" hint="查看生成的验证码"></locate>

## 模型学习

### CNN 模型

总共 5 层网络，前 3 层为卷积层，第 4、5 层为全连接层。对 4 层隐藏层都进行 dropout。网络结构如下所示：
input——>conv——>pool——>dropout——>conv——>pool——>dropout——>conv——>pool——>dropout——>fully connected layer——>dropout——>fully connected layer——>output

![](http://tensorflow-1253902462.cosgz.myqcloud.com/captcha/captcha_cnn.jpg)

**示例代码：**

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 captcha_model.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 captcha_model.py" />

```python
/// <example verb="create" file="/home/ubuntu/captcha_model.py" />
# -*- coding: utf-8 -*
import tensorflow as tf
import math

class captchaModel():
    def __init__(self,
                 width = 160,
                 height = 60,
                 char_num = 4,
                 classes = 62):
        self.width = width
        self.height = height
        self.char_num = char_num
        self.classes = classes

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def create_model(self,x_images,keep_prob):
        #first layer
        w_conv1 = self.weight_variable([5, 5, 1, 32])
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(tf.nn.bias_add(self.conv2d(x_images, w_conv1), b_conv1))
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_dropout1 = tf.nn.dropout(h_pool1,keep_prob)
        conv_width = math.ceil(self.width/2)
        conv_height = math.ceil(self.height/2)

        #second layer
        w_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_dropout1, w_conv2), b_conv2))
        h_pool2 = self.max_pool_2x2(h_conv2)
        h_dropout2 = tf.nn.dropout(h_pool2,keep_prob)
        conv_width = math.ceil(conv_width/2)
        conv_height = math.ceil(conv_height/2)

        #third layer
        w_conv3 = self.weight_variable([5, 5, 64, 64])
        b_conv3 = self.bias_variable([64])
        h_conv3 = tf.nn.relu(tf.nn.bias_add(self.conv2d(h_dropout2, w_conv3), b_conv3))
        h_pool3 = self.max_pool_2x2(h_conv3)
        h_dropout3 = tf.nn.dropout(h_pool3,keep_prob)
        conv_width = math.ceil(conv_width/2)
        conv_height = math.ceil(conv_height/2)

        #first fully layer
        conv_width = int(conv_width)
        conv_height = int(conv_height)
        w_fc1 = self.weight_variable([64*conv_width*conv_height,1024])
        b_fc1 = self.bias_variable([1024])
        h_dropout3_flat = tf.reshape(h_dropout3,[-1,64*conv_width*conv_height])
        h_fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(h_dropout3_flat, w_fc1), b_fc1))
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        #second fully layer
        w_fc2 = self.weight_variable([1024,self.char_num*self.classes])
        b_fc2 = self.bias_variable([self.char_num*self.classes])
        y_conv = tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2)

        return y_conv
```
> <checker type="output-contains" command="ls /home/ubuntu/captcha_model.py" hint="请创建 /home/ubuntu/captcha_model.py">
>    <keyword regex="/home/ubuntu/captcha_model.py" />
> </checker>

### 训练 CNN 模型

每批次采用 64 个训练样本，每 100 次循环采用 100 个测试样本检查识别准确度，当准确度大于 99% 时，训练结束，采用 GPU 需要 4-5 个小时左右，CPU 大概需要 20 个小时左右。

#### 示例代码：

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 train_captcha.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 train_captcha.py" />

```python
/// <example verb="create" file="/home/ubuntu/train_captcha.py" />
#-*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import string
import generate_captcha
import captcha_model

if __name__ == '__main__':
    captcha = generate_captcha.generateCaptcha()
    width,height,char_num,characters,classes = captcha.get_parameter()

    x = tf.placeholder(tf.float32, [None, height,width,1])
    y_ = tf.placeholder(tf.float32, [None, char_num*classes])
    keep_prob = tf.placeholder(tf.float32)

    model = captcha_model.captchaModel(width,height,char_num,classes)
    y_conv = model.create_model(x,keep_prob)
    cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_,logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    predict = tf.reshape(y_conv, [-1,char_num, classes])
    real = tf.reshape(y_,[-1,char_num, classes])
    correct_prediction = tf.equal(tf.argmax(predict,2), tf.argmax(real,2))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 1
        while True:
            batch_x,batch_y = next(captcha.gen_captcha(64))
            _,loss = sess.run([train_step,cross_entropy],feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.75})
            print ('step:%d,loss:%f' % (step,loss))
            if step % 100 == 0:
                batch_x_test,batch_y_test = next(captcha.gen_captcha(100))
                acc = sess.run(accuracy, feed_dict={x: batch_x_test, y_: batch_y_test, keep_prob: 1.})
                print ('###############################################step:%d,accuracy:%f' % (step,acc))
                if acc > 0.99:
                    saver.save(sess,"capcha_model.ckpt")
                    break
            step += 1
```
> <checker type="output-contains" command="ls /home/ubuntu/train_captcha.py" hint="请创建 /home/ubuntu/train_captcha.py">
>    <keyword regex="/home/ubuntu/train_captcha.py" />
> </checker>

**然后执行:**

```
cd /home/ubuntu;
python train_captcha.py
```

**执行结果：**

```
step:75193,loss:0.010931
step:75194,loss:0.012859
step:75195,loss:0.008747
step:75196,loss:0.009147
step:75197,loss:0.009351
step:75198,loss:0.009746
step:75199,loss:0.010014
step:75200,loss:0.009024
###############################################step:75200,accuracy:0.992500
```

#### 使用训练好的模型：
作为实验，你可以通过调整 `train_captcha.py` 文件中 `if acc > 0.99:` 代码行的准确度节省训练时间(比如将 0.99 为 0.01)，体验训练过程；我们已经通过长时间的训练得到了一个训练好的模型，可以通过如下命令将训练集下载到本地。

```
wget http://tensorflow-1253902462.cosgz.myqcloud.com/captcha/capcha_model.zip
unzip -o capcha_model.zip
```
> <checker type="output-contains" command="ls /home/ubuntu/capcha_model.ckpt.index" hint="请下载模型">
>    <keyword regex="/home/ubuntu/capcha_model.ckpt.index" />
> </checker>

### 识别验证码

#### 测试数据集：

我们在腾讯云的 COS 上准备了 100 个验证码作为测试集，使用 `wget` 命令获取：

```
wget http://tensorflow-1253902462.cosgz.myqcloud.com/captcha/captcha.zip
unzip -q captcha.zip
```
> <checker type="output-contains" command="ls /home/ubuntu/captcha.zip" hint="请下载测试数据集">
>    <keyword regex="/home/ubuntu/captcha.zip" />
> </checker>

#### 示例代码：

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 predict_captcha.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 predict_captcha.py" />

```python
/// <example verb="create" file="/home/ubuntu/predict_captcha.py" />
#-*- coding:utf-8 -*-
from PIL import Image, ImageFilter
import tensorflow as tf
import numpy as np
import string
import sys
import generate_captcha
import captcha_model

if __name__ == '__main__':
    captcha = generate_captcha.generateCaptcha()
    width,height,char_num,characters,classes = captcha.get_parameter()

    gray_image = Image.open(sys.argv[1]).convert('L')
    img = np.array(gray_image.getdata())
    test_x = np.reshape(img,[height,width,1])/255.0
    x = tf.placeholder(tf.float32, [None, height,width,1])
    keep_prob = tf.placeholder(tf.float32)

    model = captcha_model.captchaModel(width,height,char_num,classes)
    y_conv = model.create_model(x,keep_prob)
    predict = tf.argmax(tf.reshape(y_conv, [-1,char_num, classes]),2)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False,gpu_options=gpu_options)) as sess:
        sess.run(init_op)
        saver.restore(sess, "capcha_model.ckpt")
        pre_list =  sess.run(predict,feed_dict={x: [test_x], keep_prob: 1})
        for i in pre_list:
            s = ''
            for j in i:
                s += characters[j]
            print(s)
```

> <checker type="output-contains" command="ls /home/ubuntu/predict_captcha.py" hint="请创建 /home/ubuntu/predict_captcha.py">
>    <keyword regex="/home/ubuntu/predict_captcha.py" />
> </checker>

**然后执行:**

```
cd /home/ubuntu;
python predict_captcha.py captcha/0hWn.jpg
```
**执行结果：**

```
0hWn
```

## 完成实验

### 实验内容已完成

您可进行更多关于机器学习教程：

* [实验列表 - 机器学习][https://cloud.tencent.com/developer/labs/gallery?tagId=31]

关于 TensorFlow 的更多资料可参考 [TensorFlow 官网 ][https://www.tensorflow.org/]。
