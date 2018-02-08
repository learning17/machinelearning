#_*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np

class Vgg16:
    def __init__(self, X, keep_prob, num_classes = 1000,
                 train_layers=list(), weights_path='vgg16_weights.npz'):
        self.X = X
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.train_layers = train_layers
        self.weights_path = weights_path
        self.global_step = tf.Variable(1, trainable=False)

    def conv(self, X, filter_height, filter_width, num_filters,
             stride_y, stride_x, name, padding='SAME'):

        input_channels = int(X.get_shape()[-1])
        convolve = lambda x, w: tf.nn.conv2d(x, w,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('W', shape=[filter_height, filter_width,
                                                        input_channels,num_filters])
            biases = tf.get_variable('b', shape=[num_filters])
        conv = convolve(X,weights)
        out = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(out, name=scope.name)
        return relu

    def fc(self, X, num_out, name, relu=True):
        num_in = int(X.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('W',shape=[num_in,num_out])
            biases = tf.get_variable('b',[num_out])
            out = tf.nn.bias_add(tf.matmul(X,weights),biases)
        if relu:
            return tf.nn.relu(out)
        else:
            return out

    def max_pool(self, X, filter_height, filter_width,
                 stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(X, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def dropout(self, X, keep_prob, name):
        return  tf.nn.dropout(X, keep_prob, name=name)

    def create_model(self):
        #conv1_* 64 3*3
        conv1_1 = self.conv(self.X, 3, 3, 64, 1, 1, name='conv1_1')
        conv1_2 = self.conv(conv1_1,3, 3, 64, 1, 1, name='conv1_2')
        pool1 = self.max_pool(conv1_2, 2, 2, 2, 2, name='pool1')

        #conv2_* 128 3*3
        conv2_1 = self.conv(pool1, 3, 3, 128, 1, 1, name='conv2_1')
        conv2_2 = self.conv(conv2_1, 3, 3, 128, 1, 1, name='conv2_2')
        pool2 = self.max_pool(conv2_2, 2, 2, 2, 2, name='pool2')

        #conv3_* 256 3*3
        conv3_1 = self.conv(pool2, 3, 3, 256, 1, 1, name='conv3_1')
        conv3_2 = self.conv(conv3_1, 3, 3, 256, 1, 1, name='conv3_2')
        conv3_3 = self.conv(conv3_2, 3, 3, 256, 1, 1, name='conv3_3')
        pool3 = self.max_pool(conv3_3, 2, 2, 2, 2, name='pool3')

        #conv4_* 256 3*3
        conv4_1 = self.conv(pool3, 3, 3, 512, 1, 1, name='conv4_1')
        conv4_2 = self.conv(conv4_1, 3, 3, 512, 1, 1, name='conv4_2')
        conv4_3 = self.conv(conv4_2, 3, 3, 512, 1, 1, name='conv4_3')
        pool4 = self.max_pool(conv4_3, 2, 2, 2, 2, name='pool4')

        #conv5_* 256 3*3
        conv5_1 = self.conv(pool4, 3, 3, 512, 1, 1, name='conv5_1')
        conv5_2 = self.conv(conv5_1, 3, 3, 512, 1, 1, name='conv5_2')
        conv5_3 = self.conv(conv5_2, 3, 3, 512, 1, 1, name='conv5_3')
        pool5 = self.max_pool(conv5_3, 2, 2, 2, 2, name='pool5')

        #fc6 4096
        dim = int(pool5.get_shape()[-1])*int(pool5.get_shape()[-2])*int(pool5.get_shape()[-3])
        flattened = tf.reshape(pool5, [-1, dim])
        fc6 = self.fc(flattened, 4096, name='fc6')
        dropout6 = self.dropout(fc6, self.keep_prob, name='dropout6')
        #fc7 4096
        fc7 = self.fc(dropout6,4096, name='fc7')
        dropout7 = self.dropout(fc7, self.keep_prob, name='dropout7')
        #fc8 1000
        fc8 = self.fc(dropout7, self.num_classes, relu=False, name='fc8')

        return fc8

    def cross_entropy(self, logits, labels):
        with tf.name_scope("cross_entropy"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

            loss_summary = list()
            loss_summary.append(tf.summary.scalar("cross_entropy", loss))
        return loss, loss_summary

    def optimizer(self,loss, learning_rate):
        with tf.name_scope("optimizer_model"):
            tvars = [v for v in tf.trainable_variables() if v.name.split('/')[0] in self.train_layers]
            clipped_gradient, gradient_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(zip(clipped_gradient, tvars),global_step=self.global_step)

            gradient_summary = list()
            gradient_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradient)))
            gradient_summary.append(tf.summary.scalar("gradient_norm", gradient_norm))
            gradient_summary.append(tf.summary.scalar("learning_rate",learning_rate))
        return train_op,gradient_summary

    def accuracy(self, logits, labels):
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            accuracy_summary = list()
            accuracy_summary.append(tf.summary.scalar("accuracy", accuracy))
        return accuracy, accuracy_summary

    def initial_weights(self,sess):
        """
        http://www.cs.toronto.edu/~frossard/post/vgg16/
        """
        weights = np.load(self.weights_path)
        keys = sorted(weights.keys())
        for key in keys:
            pos = key.rfind('_')
            lay, para_key = key[:pos], key[pos+1:]
            if lay not in self.train_layers:
                with tf.variable_scope(lay, reuse=True):
                    para_value = tf.get_variable(para_key, trainable=False)
                    value = weights[key]
                    print(key,value.shape)
                    sess.run(para_value.assign(value))
