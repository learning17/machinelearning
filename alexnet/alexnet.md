# TensorFlow - 基于 AlexNet 图像分类

## Architecture of the Network

![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_1.png)

AlexNet 是具有历史意义的一个网络结构，在此之前，深度学习已经沉寂了很久。2012 年，AlexNet 在当年的 ImageNet 图像分类竞赛中，top-5 错误率比上一年的冠军下降了十个百分点，而且远远超过当年的第二名。网络结构如图 1，包含五层卷积层和三层全连接层，AlexNet 之所以能够成功，原因是：
* `非线性激活函数`：ReLU；
* `防止过拟合`：Data Augmentation, Dropout；
* `大数据训练`：百万级 ImageNet 图像数据；
* `局部归一化`：LRN 归一化层；
* `其他`：分 Group 实现双 GPU 并行；

### ReLU

![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_2.png)
$$f(x)=max(0,z)$$
优点：
* 在输入大于 0 时候，不会出现梯度消失；
* 相较于 sigmoid 和 tanh 收敛速度大大提升，不含有 exp 运算，计算速度提升。

缺点：
* 输出不是关于中心点对称；
* 当输入小于 0，在前向传播时 ReLU 一直处于非激活状态，那么反向传播就会出现梯度消失(未激活的神经元局部梯度为 0 )，因此非激活的神经元无法进行反向传播，它的权重也不会更新；当输入大于 0 ，ReLU 神经元直接将梯度传给前面层网络；当输入为 0，梯度未定义，我们可以简单指定为 0 或 1（实际出现输入为 0 的概率很小）；
* 可能出现 ReLU 神经元失活问题，如 $max(0,w^{T}x+b)$ ，假设 $w、b$ 都初始化为 0，或者更新过程中，$w、b$ 更新为一种特别状态，使得 永远小于 0，那么该神经元权重将永远得不到更新。

### Data Augmentation

神经网络是靠数据喂出来的，同时扩大数据集合可以有效地解决过拟合问题，可以通过一些简单的变换从已有的训练数据集中生成一些新的数据，来扩充训练数据集。
* `镜像对称（Mirroring）`

![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_3.png)

* `随机裁剪（Random Cropping）`

![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_4.png)

* `色彩变换（Color shifting）`

在时间中增加的 RGB 值是根据某种概率分布来决定的。PCA 颜色增强，其含义是：比如图像呈现紫色，即主要含有红色和蓝色，绿色很少，那么 PCA 颜色增强就会对红色和蓝色增减很多，绿色变化相对少一点，所以使总体的颜色保持一致。
![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_5.png)

AlexNet Data Augmentation 做法是：
* `Random Cropping`：从 $256\times256$ 的图像中随机裁剪 $224\times224$（包括 Mirroring 图像），相当于将样本增加 $(256-224)^{2}\times2=2048$；
* `Color shifting`：用PCA方法。将每个像素的 RGB 值 $\left[ I^{R}_{xy},I^{G}_{xy},I^{B}_{xy}\right]^{T}$ 加上 $\left[p_{1},p_{2},p_{3}\right]\left[ \alpha_{1}\lambda_{1},\alpha_{2}\lambda_{2},\alpha_{3}\lambda_{3}\right]^{T}$ 其中 $p_{i}$ 和 $\lambda_{i}$ 分别是 RGB 值的 $3\times3$ 协方差矩阵的第 $i$ 个特征向量和特征值。$\alpha_{i}$ 是一个服从均值为 $0$，标准差为 $0.1$ 的高斯分布的高斯变量。

### Dropout

![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_7.png)
前向传播的时候，让某个神经元的激活值以一定的概率 $p$ 停止工作（即把该神经元的激活值设置为0）；为了不影响整个网络输出的期望值，未被 Dropout 的神经元的激活值除以 $1-p$。

AlexNet 选择概率为 $0.5$，在第 1、2 个全连接层用了 Dropout 技术，较少过拟合。

### 局部归一化
$b^{i}_{x,y} = a^{i}_{x,y}/\left(k+\alpha\sum^{min\left(N-1,i+n/2\right)}_{j=max\left(0,i-n/2\right)}\left(a^{j}_{x,y}\right)^{2}  \right)^{\beta}$
其中 $a^{i}_{x,y}$ 表示第 $i$ 个卷积核在 $(x,y)$ 位置产生的值再应用 ReLU 激活函数后的结果，$n$ 表示相邻的几个卷积核，$N$ 表示这一层总的卷积核数量，$k=2,n=5,\alpha=0.0001,\beta=0.75$ 是超参数。

### 网络结构分析

![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_6.png)

#### 第一层 卷积层

* `conv`：输入 $227\times227\times3$，卷积参数 $11\times11\times96$ 卷积核，步长为 $strides=4$，$Valid$ 方式。输出$55\times55\times96$，其中$\left\lfloor\frac{227-11}{4}+1\right\rfloor=55$ 。
* `lrn`：对卷积结果进行局部归一化。
* `max-pool`：输入 $55\times55\times96$，池化参数 $z=3,s=2$，$Valid$ 方式，输出 $27\times27\times96$，其中$\left\lfloor\frac{55-3}{2}+1\right\rfloor=27$

#### 第二层 卷积层

* `conv`：输入 $27\times27\times96$，卷积参数 $5\times5\times256$ 卷积核，步长为 $strides=1$，$Same$ 方式。输出 $27\times27\times256$，其中 $padding：\frac{5-1}{2}=2$，$\left\lfloor\frac{27+2\times2-5}{1}+1\right\rfloor=27$。
* `lrn`：对卷积结果进行局部归一化。
* `max-pool`：输入 $27\times27\times256$，池化参数 $z=3,s=2$，$Valid$ 方式，输出 $13\times13\times256$，其中$\left\lfloor\frac{27-3}{2}+1\right\rfloor=13$

#### 第三层 卷积层

* `conv`：输入 $13\times13\times256$，卷积参数 $3\times3\times384$ 卷积核，步长为 $strides=1$，$Same$ 方式。输出 $13\times13\times384$，其中 $padding：\frac{3-1}{2}=1$，$\left\lfloor\frac{13+2\times1-3}{1}+1\right\rfloor=13$。

#### 第四层 卷积层

* `conv`：输入 $13\times13\times384$，卷积参数 $3\times3\times384$ 卷积核，步长为 $strides=1$，$Same$ 方式。输出 $13\times13\times384$，其中 $padding：\frac{3-1}{2}=1$，$\left\lfloor\frac{13+2\times1-3}{1}+1\right\rfloor=13$。

#### 第五层 卷积层

* `conv`：输入 $13\times13\times384$，卷积参数 $3\times3\times256$ 卷积核，步长为 $strides=1$，$Same$ 方式。输出 $13\times13\times256$，其中 $padding：\frac{3-1}{2}=1$，$\left\lfloor\frac{13+2\times1-3}{1}+1\right\rfloor=13$。
* `max-pool`：输入  $13\times13\times256$，池化参数 $z=3,s=2$，$Valid$ 方式，输出 $6\times6\times256$，其中$\left\lfloor\frac{13-3}{2}+1\right\rfloor=6$

#### 第六层 全连接层
* `fc`：$6\times6\times256\rightarrow4096$。
* `Dropout`：0.5。

#### 第七层 全连接层
* `fc`：$4096\rightarrow4096$。
* `Dropout`：0.5。

#### 第八层 全连接层
* `fc`：$4096\rightarrow1000$。

## 代码分析

### 代码结构
* `generate_alexnet.py`：训练数据；
* `alexnet.py`：AlexNet 模型；
* `train_alexnet.py`：训练 AlexNet 模型；
* `predict_alexnet.py`：验证 AlexNet 模型。

### 训练数据

采用  [Kaggle Dogs vs. Cats Redux Competition][https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data] 作为训练数据，训练猫、狗二分类器。把train.zip数据分为训练集和验证集 (70/30) ，并分别保存在 `train.txt` 和 `val.txt`。格式为：filename  label，0：Dogs，1：Dogs。`get_image_path_label` 实现该功能。 
```
data/train/dog.7412.jpg 1                               
data/train/cat.1290.jpg 0
```
### AlexNet 模型

* 1st layer: conv -> lrn -> pool
* 2nd layer: conv -> lrn -> pool (split into 2 groups)
* 3rd layer: conv
* 4th layer: conv (split into 2 groups)
* 5th layer: conv -> pool (split into 2 groups)
* 6th layer: fully-connected -> dropout
* 7th layer: fully-connected -> dropout
* 8th layer: fully-connected

![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_10.png)

### 训练 AlexNet 模型

本次实验没有从头开始完全训练整个网络，前面五层的卷积层直接采用已训练好的[参数][http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/] ，我们训练后面三层的全连接层。
在验证集上的准确度为 $0.966$ 。
![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_8.png)
![](https://deeplearning-1254072688.cos.ap-guangzhou.myqcloud.com/alexnet/alexnet_9.png)

### 验证 AlexNet 模型

主要实现两个功能：
* ImageNet 图像分类，取概率最大的五类；
* Dogs vs. Cats 分类。
