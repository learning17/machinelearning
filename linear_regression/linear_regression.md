# tensorflow — 线性回归
## tensorflow实现线性回归模型代码
### 前期准备

tensorflow相关API可以到“tensorflow — 相关API”教程学习
### 模型构建

**示例代码：**

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 linear_regression_model.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 linear_regression_model.py" />

```python
/// <example verb="create" file="/home/ubuntu/linear_regression_model.py" />
#!/usr/bin/python
# -*- coding: utf-8 -*

import tensorflow as tf
import numpy as np

class linearRegressionModel:

  def __init__(self,x_dimen):
    self.x_dimen = x_dimen
    self._index_in_epoch = 0
    self.constructModel()
    self.sess = tf.Session()
    self.sess.run(tf.global_variables_initializer())

  #权重初始化
  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

  #偏置项初始化
  def bias_variable(self,shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

  #每次选取100个样本，如果选完，重新打乱
  def next_batch(self,batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_datas:
        perm = np.arange(self._num_datas)
        np.random.shuffle(perm)
        self._datas = self._datas[perm]
        self._labels = self._labels[perm]
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_datas
    end = self._index_in_epoch
    return self._datas[start:end],self._labels[start:end]

  def constructModel(self):
    self.x = tf.placeholder(tf.float32, [None,self.x_dimen])
    self.y = tf.placeholder(tf.float32,[None,1])
    self.w = self.weight_variable([self.x_dimen,1])
    self.b = self.bias_variable([1])
    self.y_prec = tf.nn.bias_add(tf.matmul(self.x, self.w), self.b)

    mse = tf.reduce_mean(tf.squared_difference(self.y_prec, self.y))
    l2 = tf.reduce_mean(tf.square(self.w))
    self.loss = mse + 0.15*l2
    self.train_step = tf.train.AdamOptimizer(0.1).minimize(self.loss)

  def train(self,x_train,y_train,x_test,y_test):
    self._datas = x_train
    self._labels = y_train
    self._num_datas = x_train.shape[0]
    for i in range(5000):
        batch = self.next_batch(100)
        self.sess.run(self.train_step,feed_dict={self.x:batch[0],self.y:batch[1]})
        if i%10 == 0:
            train_loss = self.sess.run(self.loss,feed_dict={self.x:batch[0],self.y:batch[1]})
            print('step %d,test_loss %f' % (i,train_loss))

  def predict_batch(self,arr,batch_size):
    for i in range(0,len(arr),batch_size):
        yield arr[i:i + batch_size]

  def predict(self, x_predict):
    pred_list = []
    for x_test_batch in self.predict_batch(x_predict,100):
      pred = self.sess.run(self.y_prec, {self.x:x_test_batch})
      pred_list.append(pred)
    return np.vstack(pred_list)
```
> <checker type="output-contains" command="ls /home/ubuntu/linear_regression_model.py" hint="请创建并执行 /home/ubuntu/linear_regression_model.py">
>    <keyword regex="/home/ubuntu/linear_regression_model.py" />
> </checker>

### 训练模型并和sklearn库线性回归模型对比

**示例代码：**

现在您可以在 [/home/ubuntu][create-example] 目录下[创建源文件 run.py][create-example]，内容可参考：

> <locate for="create-example" path="/home/ubuntu" verb="create" hint="右击创建 run.py" />

```python
/// <example verb="create" file="/home/ubuntu/run.py" />
#!/usr/bin/python
# -*- coding: utf-8 -*

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from linear_regression_model import linearRegressionModel as lrm

if __name__ == '__main__':
    x, y = make_regression(7000)
    x_train,x_test,y_train, y_test = train_test_split(x, y, test_size=0.5)
    y_lrm_train = y_train.reshape(-1, 1)
    y_lrm_test = y_test.reshape(-1, 1)

    linear = lrm(x.shape[1])
    linear.train(x_train, y_lrm_train,x_test,y_lrm_test)
    y_predict = linear.predict(x_test)
    print("Tensorflow R2: ", r2_score(y_predict.ravel(), y_lrm_test.ravel()))

    lr = LinearRegression()
    y_predict = lr.fit(x_train, y_train).predict(x_test)
    print("Sklearn R2: ", r2_score(y_predict, y_test)) #采用r2_score评分函数
```
然后执行:

```
cd /home/ubuntu;
python run.py
```
**执行结果：**

```
step 2410,test_loss 26.531937
step 2420,test_loss 26.542793
step 2430,test_loss 26.533974
step 2440,test_loss 26.530540
step 2450,test_loss 26.551474
step 2460,test_loss 26.541542
step 2470,test_loss 26.560783
step 2480,test_loss 26.538080
step 2490,test_loss 26.535666
('Tensorflow R2: ', 0.99999612588302389)
('Sklearn R2: ', 1.0)
```
> <checker type="output-contains" command="ls /home/ubuntu/run.py" hint="请创建并执行 /home/ubuntu/run.py">
>    <keyword regex="/home/ubuntu/run.py" />
> </checker>

## 完成实验

### 实验内容已完成

您可进行更多关于机器学习教程：

* [实验列表 - 机器学习][https://cloud.tencent.com/developer/labs/gallery?tagId=31]

关于 TensorFlow 的更多资料可参考 [TensorFlow 官网 ][https://www.tensorflow.org/]。