TensorFlow - 基于 GANs 生成人脸
============================

## 简介

给定一批样本，基于 TensoFlow 训练 GANs 网络，能够生成类似的新样本，本教程主要参考 Brandon Amos 的 Image Completion 博客，GANs 网络包含 generator 网络（随机信号 z 作为输入，生成人脸图片）和 discriminator 网络（判断图片是否是人脸）。

### 步骤简介
本教程一共分为四个部分
* `generate_face.py` - 读取人脸训练数据、产生随机数；
* `gan_model.py` - GANs 网络模型；
* `train_gan.py` - 训练 GANs 网络模型；
* `predict_gan.py` - 生成人脸。

## 数据学习

### 获取训练数据
我们在腾讯云的 COS 上准备了 [CelebA][http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html] 训练数据，使用 `wget` 命令获取：
```
wget http://tensorflow-1253902462.cosgz.myqcloud.com/gan_face/img_align_celeba.zip
unzip -q img_align_celeba.zip
```
> <checker type="output-contains" command="ls /home/ubuntu/img_align_celeba.zip" hint="请使用 wget 下载训练数据">
>   <keyword regex="/home/ubuntu/img_align_celeba.zip" />
> </checker>

### 数据预处理
安装依赖库
```
pip install scipy
pip install pillow
```
#### 处理思路：
* 原始图片大小为 218 x 178 ，从中间裁剪 108 x 108 区域，然后缩小为 64 x 64。
* 生成维度为 100 服从正态分布的随机向量，作为 generator 网络的输入，生成新的人脸图片。

#### 示例代码：
现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 generate_face.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 generate_face.py" />

```python
/// <example verb="create" file="/home/ubuntu/generate_face.py" />
#-*- coding:utf-8 -*-
import itertools
import os
from glob import glob
import numpy as np
import scipy.misc
import tensorflow as tf


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img

def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

def inverse_transform(images):
    return (images+1.)/2.

def save_images(images,size,image_path):
    return imsave(inverse_transform(images), size, image_path)

class generateFace:
    def __init__(self,hparams):
        self.formats = ["png","jpg","jpeg"]
        self.datas_path = self.get_datas_path(hparams.data_root)
        self.datas_size = len(self.datas_path)
        self.crop_h = hparams.crop_h
        self.crop_w = hparams.crop_w
        self.resize_h = hparams.resize_h
        self.resize_w = hparams.resize_w
        self.is_crop = hparams.is_crop
        self.z_dim = hparams.z_dim
        self._index_in_epoch = 0

    def get_datas_path(self,data_root):
        return list(itertools.chain.from_iterable(
            glob(os.path.join(data_root,"*.{}".format(ext))) for ext in self.formats))

    def get_image(self,path):
        img = scipy.misc.imread(path,mode='RGB').astype(np.float)
        if(self.is_crop): #截取中间部分
            h,w = img.shape[:2] #图像宽、高
            assert(h > self.crop_h and w > self.crop_w)
            j = int(round((h - self.crop_h)/2.))
            i = int(round((w - self.crop_w)/2.))
            img = img[j:j+self.crop_h,i:i+self.crop_w]
        img = scipy.misc.imresize(img,[self.resize_h,self.resize_w])
        return np.array(img)/127.5 - 1.

    def get_batch(self,batch_files):
        batch_images = [self.get_image(path) for path in batch_files]
        batch_images = np.array(batch_images).astype(np.float32)
        batch_z = np.random.uniform(-1,1,size=(len(batch_files),self.z_dim))
        return batch_images,batch_z

    def get_sample(self,sample_size):
        assert(self.datas_size > sample_size)
        np.random.shuffle(self.datas_path)
        sample_files = self.datas_path[0:sample_size]
        return self.get_batch(sample_files)

    def next_batch(self,batch_size):
        assert(self.datas_size > batch_size)
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if(self._index_in_epoch > self.datas_size):
            np.random.shuffle(self.datas_path)
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        batch_files = self.datas_path[start:end]
        return self.get_batch(batch_files)

```
> <checker type="output-contains" command="ls /home/ubuntu/generate_face.py" hint="请创建并执行 /home/ubuntu/generate_face.py">
>    <keyword regex="/home/ubuntu/generate_face.py" />
> </checker>

#### 生成数据：
我们可以直观感受下生成的数据。可以在终端中一步一步执行下面命令：

* 启动 python：

```
cd /home/ubuntu/
python
from generate_face import *
import tensorflow as tf
```
* 初始化 generate_face

```
hparams = tf.contrib.training.HParams(
    data_root = './img_align_celeba',
    crop_h = 108,
    crop_w = 108,
    resize_h = 64,
    resize_w = 64,
    is_crop = True,
    z_dim = 100,
    batch_size = 64,
    sample_size = 64,
    output_h = 64,
    output_w = 64,
    gf_dim = 64,
    df_dim = 64)
face = generateFace(hparams)
```
* 查看处理后的人脸数据和随机数据 z

```
img,z = face.next_batch(1)
z
save_images(img,(1,1),"test.jpg")
```

## 模型学习

### GANs 模型
* generator 网络：五层网络，采用反卷积，从 100 维的 z 信号生成人脸图片，网络结构见下图：

![](http://tensorflow-1253902462.cosgz.myqcloud.com/gan_face/generator.png)

* discriminator 网络：是一个五层的判别网络，网络结构见下图：

![](http://tensorflow-1253902462.cosgz.myqcloud.com/gan_face/discriminator.png)


#### 示例代码：
现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 gan_model.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 gan_model.py" />

```python
/// <example verb="create" file="/home/ubuntu/gan_model.py" />
#-*- coding:utf-8 -*-
import tensorflow as tf
import math

class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name
    def __call__(self,x,train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)
class ganModel:
    def __init__(self,hparams):
        self.batch_size = hparams.batch_size
        self.gf_dim = hparams.gf_dim
        self.df_dim = hparams.df_dim
        self.output_h = hparams.output_h
        self.output_w = hparams.output_w
        #batch normalization
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.global_step = tf.Variable(1, trainable=False)

    def linear(self,input_z,output_size,scope=None, stddev=0.02, bias_start=0.0):
        shape = input_z.get_shape().as_list()
        with tf.variable_scope(scope or "linear"):
            matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                     tf.random_normal_initializer(stddev=stddev))
            bias = tf.get_variable("bias", [output_size],
                                   initializer=tf.constant_initializer(bias_start))
            return tf.matmul(input_z,matrix) + bias

    def conv2d_transpose(self,input_, output_shape,
                         k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                         name="conv2d_transpose"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, d_h, d_w, 1])
            biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, biases)
            return deconv

    def conv2d(self,image,output_dim,
               k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
               name="conv2d"):
        with tf.variable_scope(name):
            w = tf.get_variable('w', [k_h, k_w, image.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))
            conv = tf.nn.conv2d(image, w, strides=[1, d_h, d_w, 1], padding='SAME')
            biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)
            return conv

    def lrelu(self,x, leak=0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def conv_out_size_same(self,size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def generator(self,z,is_training):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_h, self.output_w  #64*64
            s_h2, s_w2 = self.conv_out_size_same(s_h, 2), self.conv_out_size_same(s_w, 2) #32*32
            s_h4, s_w4 = self.conv_out_size_same(s_h2, 2), self.conv_out_size_same(s_w2, 2) #16*16
            s_h8, s_w8 = self.conv_out_size_same(s_h4, 2), self.conv_out_size_same(s_w4, 2) #8*8
            s_h16, s_w16 = self.conv_out_size_same(s_h8, 2), self.conv_out_size_same(s_w8, 2) #4*4

            z_ = self.linear(z,self.gf_dim*8*s_h16*s_w16,'g_h0_lin')
            h0 = tf.reshape(z_,[-1,s_h16,s_w16,self.gf_dim*8])
            h0 = tf.nn.relu(self.g_bn0(h0,is_training))

            h1 = self.conv2d_transpose(h0,[self.batch_size,s_h8,s_w8,self.gf_dim*4],name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1,is_training))

            h2 = self.conv2d_transpose(h1,[self.batch_size,s_h4,s_w4,self.gf_dim*2],name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2,is_training))

            h3 = self.conv2d_transpose(h2,[self.batch_size,s_h2,s_w2,self.gf_dim*1],name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3,is_training))

            h4 = self.conv2d_transpose(h3,[self.batch_size,s_h,s_w,3],name='g_h4')

            return tf.nn.tanh(h4)

    def discriminator(self,image,is_training,reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = self.lrelu(self.conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = self.lrelu(self.d_bn1(self.conv2d(h0, self.df_dim*2, name='d_h1_conv'), is_training))
            h2 = self.lrelu(self.d_bn2(self.conv2d(h1, self.df_dim*4, name='d_h2_conv'), is_training))
            h3 = self.lrelu(self.d_bn3(self.conv2d(h2, self.df_dim*8, name='d_h3_conv'), is_training))
            h3 = tf.reshape(h3,[-1,8192]) #8192 = self.df_dim*8*4*4
            h4 = self.linear(h3,1,'d_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def build_model(self,is_training,images,z):
        z_sum = tf.summary.histogram("z",z)

        G = self.generator(z,is_training)
        D,D_logits = self.discriminator(images,is_training)
        D_,D_logits_ = self.discriminator(G,is_training,reuse=True)

        d_sum = tf.summary.histogram("d",D)
        d__sum = tf.summary.histogram("d_",D_)
        G_sum = tf.summary.image("G", G)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits,
                                                    labels=tf.ones_like(D)))#对于discriminator，尽量判断images是货真价实
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,
                                                    labels=tf.zeros_like(D_)))#对于discriminator，尽量判断G是伪冒

        d_loss_real_sum = tf.summary.scalar("d_loss_real",d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake",d_loss_fake)

        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_,
                                                    labels=tf.ones_like(D_)))#对于generator，尽量然D判断G是货真价实的
        d_loss = d_loss_real + d_loss_fake

        g_loss_sum = tf.summary.scalar("g_loss", g_loss)
        d_loss_sum = tf.summary.scalar("d_loss", d_loss)

        t_vars = tf.trainable_variables() #discriminator、generator两个网络参数分开训练
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        g_sum = tf.summary.merge([z_sum,d__sum,G_sum,d_loss_fake_sum,g_loss_sum])
        d_sum = tf.summary.merge([z_sum,d_sum,d_loss_real_sum,d_loss_sum])

        return g_loss,d_loss,g_vars,d_vars,g_sum,d_sum,G

    def optimizer(self,g_loss,d_loss,g_vars,d_vars,learning_rate = 0.0002,beta1=0.5):
        d_optim = tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(d_loss,global_step=self.global_step,var_list=d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate,beta1=beta1).minimize(g_loss,var_list=g_vars)
        return d_optim,g_optim
```
> <checker type="output-contains" command="ls /home/ubuntu/gan_model.py" hint="请创建并执行 /home/ubuntu/gan_model.py">
>    <keyword regex="/home/ubuntu/gan_model.py" />
> </checker>

### 训练 GANs 模型
训练 13 万次后，损失函数基本保持不变，单个 GPU 大概需要 6 个小时左右，如果采用 CPU 大概需要 1 天半的时间，你可以调整循环次数，体验下训练过程，可以直接下载我们训练好的模型。

#### 示例代码：
现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 train_gan.py][create-example]，内容可参考：
> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 train_gan.py" />

```python
/// <example verb="create" file="/home/ubuntu/train_gan.py" />
#-*- coding:utf-8 -*-
from generate_face import *
from gan_model import ganModel
import tensorflow as tf

if __name__ == '__main__':
    hparams = tf.contrib.training.HParams(
        data_root = './img_align_celeba',
        crop_h = 108,    #对原始图片裁剪后高
        crop_w = 108,    #对原始图片裁剪后宽
        resize_h = 64,   #对裁剪后图片缩放的高
        resize_w = 64,   #对裁剪图片缩放的宽
        is_crop = True,  #是否裁剪
        z_dim = 100,     #随机噪声z的维度，用户generator生成图片
        batch_size = 64, #批次
        sample_size = 64,#选取作为测试样本
        output_h = 64,   #generator生成图片的高
        output_w = 64,   #generator生成图片的宽
        gf_dim = 64,     #generator的feature map的deep
        df_dim = 64)     #discriminator的feature map的deep
    face = generateFace(hparams)
    sample_images,sample_z = face.get_sample(hparams.sample_size)
    is_training = tf.placeholder(tf.bool,name='is_training')
    images = tf.placeholder(tf.float32, [None,hparams.resize_h,hparams.output_w,3],name='real_images')
    z = tf.placeholder(tf.float32, [None,hparams.z_dim], name='z')
    model = ganModel(hparams)
    g_loss,d_loss,g_vars,d_vars,g_sum,d_sum,G = model.build_model(is_training,images,z)
    d_optim,g_optim = model.optimizer(g_loss,d_loss,g_vars,d_vars)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./ckpt')
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter("train_gan", sess.graph)
        step = 0
        while True:
            step = model.global_step.eval()
            batch_images,batch_z = face.next_batch(hparams.batch_size)
            #Update D network
            _, summary_str = sess.run([d_optim,d_sum],
                                           feed_dict={images:batch_images, z:batch_z, is_training:True})
            summary_writer.add_summary(summary_str,step)

            #Update G network
            _, summary_str = sess.run([g_optim,g_sum],
                                           feed_dict={z:batch_z, is_training:True})
            summary_writer.add_summary(summary_str,step)

            d_err = d_loss.eval({images:batch_images, z:batch_z, is_training:False})
            g_err = g_loss.eval({z:batch_z,is_training:False})
            print("step:%d,d_loss:%f,g_loss:%f" % (step,d_err,g_err))
            if step%1000 == 0:
                samples, d_err, g_err = sess.run([G,d_loss,g_loss],
                                                   feed_dict={images:sample_images, z:sample_z, is_training:False})
                print("sample step:%d,d_err:%f,g_err:%f" % (step,d_err,g_err))
                save_images(samples,image_manifold_size(samples.shape[0]), './samples/train_{:d}.png'.format(step))
                saver.save(sess,"./ckpt/gan.ckpt",global_step = step)
```
> <checker type="output-contains" command="ls /home/ubuntu/train_gan.py" hint="请创建并执行 /home/ubuntu/train_gan.py">
>    <keyword regex="/home/ubuntu/train_gan.py" />
> </checker>

**然后执行:**
```
cd /home/ubuntu;
python train_gan.py
```
**执行结果：**
```
step:1,d_loss:1.276464,g_loss:0.757655
step:2,d_loss:1.245563,g_loss:0.916217
step:3,d_loss:1.253453,g_loss:1.111729
step:4,d_loss:1.381798,g_loss:1.408796
step:5,d_loss:1.643821,g_loss:1.928348
step:6,d_loss:1.770768,g_loss:2.165831
step:7,d_loss:2.172084,g_loss:2.746789
step:8,d_loss:2.192665,g_loss:3.120509
```
**下载已有模型:**
```
wget http://tensorflow-1253902462.cosgz.myqcloud.com/gan_face/GANs_model.zip
unzip -o GANs_model.zip
```
> <checker type="output-contains" command="ls /home/ubuntu/GANs_model.zip" hint="下载 GANs_model.zip">
>    <keyword regex="/home/ubuntu/GANs_model.zip" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/gan.ckpt-130000.data-00000-of-00001" hint="请解压 GANs_model.zip">
>    <keyword regex="/home/ubuntu/gan.ckpt-130000.data-00000-of-00001" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/gan.ckpt-130000.index" hint="请解压 GANs_model.zip">
>    <keyword regex="/home/ubuntu/gan.ckpt-130000.index" />
> </checker>

> <checker type="output-contains" command="ls /home/ubuntu/gan.ckpt-130000.meta" hint="请解压 GANs_model.zip">
>    <keyword regex="/home/ubuntu/gan.ckpt-130000.meta" />
> </checker>

### 生成人脸

利用训练好的模型，我们可以开始生成人脸。

**示例代码：**

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 predict_gan.py][create-example]，内容可参考：
> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 predict_gan.py" />

```python
/// <example verb="create" file="/home/ubuntu/predict_gan.py" />
#-*- coding:utf-8 -*-
from generate_face import *
from gan_model import ganModel
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    hparams = tf.contrib.training.HParams(
        z_dim = 100,
        batch_size = 1,
        gf_dim = 64,
        df_dim = 64,
        output_h = 64,
        output_w = 64)

    is_training = tf.placeholder(tf.bool,name='is_training')
    z = tf.placeholder(tf.float32, [None,hparams.z_dim], name='z')
    sample_z = np.random.uniform(-1,1,size=(hparams.batch_size,hparams.z_dim))
    model = ganModel(hparams)
    G = model.generator(z,is_training)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,"gan.ckpt-130000")
        samples = sess.run(G,feed_dict={z:sample_z,is_training:False})
        save_images(samples,image_manifold_size(samples.shape[0]),'face.png')
        print("done")

```
> <checker type="output-contains" command="ls /home/ubuntu/predict_gan.py" hint="请创建并执行 /home/ubuntu/predict_gan.py">
>    <keyword regex="/home/ubuntu/predict_gan.py" />
> </checker>

**然后执行:**

```
cd /home/ubuntu
python predict_gan.py
```
**执行结果：**

现在您可以在 [查看 /home/ubuntu/face.png][edit-host]
> <edit for="edit-host" file="/home/ubuntu/face.png" />

## 完成实验

### 实验内容已完成

您可进行更多关于机器学习教程：

* [实验列表 - 机器学习][https://cloud.tencent.com/developer/labs/gallery?tagId=31]

关于 TensorFlow 的更多资料可参考 [TensorFlow 官网 ][https://www.tensorflow.org/]。