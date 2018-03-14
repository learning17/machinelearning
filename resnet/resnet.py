# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

class Resnet:
    def __init__(self, is_training, bottleneck = False, num_classes = 1000, num_blocks = [3, 4, 6 ,3]):
        self.is_training = is_training
        self.bottleneck = bottleneck
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.global_step = tf.Variable(1, trainable=False)

    def _get_variable(self, shape, name = "W",initializer=None,
                      weight_decay=0.0, dtype=tf.float32, trainable=True):
        if weight_decay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
        else:
            regularizer = None
        if initializer == None:
            initializer = tf.truncated_normal_initializer(stddev=0.1)
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               dtype=dtype,
                               regularizer=regularizer,
                               trainable=trainable)

    def conv(self, X, filter_height, filter_width, out_channels,
             stride_y, stride_x, name, padding='SAME'):
        in_channels = int(X.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = self._get_variable(shape = [filter_height, filter_width,in_channels,out_channels],
                                         weight_decay = 0.00004)

        return tf.nn.conv2d(X, weights, strides=[1, stride_y, stride_x, 1],padding=padding)

    def max_pool(self, X, filter_height, filter_width,
                 stride_y, stride_x, name="pool", padding='SAME'):
        return tf.nn.max_pool(X, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def fc(self, X, num_out, name):
        num_in = int(X.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = self._get_variable(shape = [num_in, num_out],
                                         weight_decay = 0.00004)
            biases = self._get_variable(shape = [num_out],
                                        name = "b",
                                        initializer=tf.zeros_initializer)
        return tf.nn.bias_add(tf.matmul(X,weights),biases)

    def bn(self, X, name, eps=0.001, decay=0.9997):
        x_shape = X.get_shape()
        params_shape = x_shape[-1:]
        axis = list(range(len(x_shape) - 1))
        with tf.variable_scope(name) as scope:
            beta = self._get_variable(shape = params_shape,
                                      name = "beta",
                                      initializer=tf.zeros_initializer)
            gamma = self._get_variable(shape = params_shape,
                                       name = "gamma",
                                       initializer=tf.ones_initializer)
            moving_mean = self._get_variable(shape = params_shape,
                                             name = "moving_mean",
                                             initializer=tf.zeros_initializer,
                                             trainable = False)
            moving_variance = self._get_variable(shape = params_shape,
                                                 name = "moving_variance",
                                                 initializer=tf.ones_initializer,
                                                 trainable = False)
        def mean_var_with_update():
            mean, variance = tf.nn.moments(X, axis)
            with tf.control_dependencies([moving_averages.assign_moving_average(moving_mean, mean, decay),
                                          moving_averages.assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(self.is_training, mean_var_with_update, lambda: (moving_mean, moving_variance))
        X = tf.nn.batch_normalization(X, mean, variance, beta, gamma, eps)
        return X

    def block(self, X, out_channels, stride = 1):
        in_channels = int(X.get_shape()[-1])
        shortcut = X
        final_out_channels = out_channels

        if self.bottleneck:
            final_out_channels = 4*out_channels
            with tf.variable_scope('a') as scope:
                X = self.conv(X, 1, 1, out_channels, stride, stride, scope)
                X = self.bn(X,scope)
                X = tf.nn.relu(X)
            with tf.variable_scope('b') as scope:
                X = self.conv(X, 3, 3, out_channels, 1, 1, scope)
                X = self.bn(X,scope)
                X = tf.nn.relu(X)
            with tf.variable_scope('c') as scope:
                X = self.conv(X, 1, 1, final_out_channels, 1, 1, scope)
                X = self.bn(X,scope)
        else:
            with tf.variable_scope('A') as scope:
                X = self.conv(X, 3, 3, out_channels, stride, stride, scope)
                X = self.bn(X,scope)
                X = tf.nn.relu(X)
            with tf.variable_scope('B') as scope:
                X = self.conv(X, 3, 3, final_out_channels, 1, 1, scope)
                X = self.bn(X,scope)

        with tf.variable_scope('shortcut') as scope:
            if out_channels != in_channels or stride != 1:
                shortcut = self.conv(shortcut, 1, 1, final_out_channels, stride, stride, scope)
                shortcut = self.bn(shortcut, scope)
        return tf.nn.relu(X + shortcut)

    def stack(self, X, num_blocks, out_channels, stride=1):
        for n in range(num_blocks):
            s = stride if n == 0 else 1
            with tf.variable_scope('block_%d' % (n + 1)):
                X = self.block(X, out_channels, s)
        return X

    def create_model(self, X):
        with tf.variable_scope('scale1') as scope:
            X = self.conv(X, 7, 7, 64, 2, 2, scope)
            X = self.bn(X,scope)
            X = tf.nn.relu(X)
            X = self.max_pool(X, 3, 3, 2, 2)

        with tf.variable_scope('scale2'):
            X = self.stack(X, self.num_blocks[0], 64, 1)

        with tf.variable_scope('scale3'):
            X = self.stack(X, self.num_blocks[1], 128, 2)

        with tf.variable_scope('scale4'):
            X = self.stack(X, self.num_blocks[2], 256, 2)

        with tf.variable_scope('scale5'):
            X = self.stack(X, self.num_blocks[3], 512, 2)

        X = tf.reduce_mean(X, axis=[1, 2], name="avg_pool")

        with tf.variable_scope('fc') as scope:
            X = self.fc(X, self.num_classes, scope)
        return X

    def cross_entropy(self, logits, labels):
        with tf.name_scope("cross_entropy"):
            cross_entropy_mean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n([cross_entropy_mean] + regularization_losses)

            loss_summary = list()
            loss_summary.append(tf.summary.scalar("loss", loss))
        return loss, loss_summary

    def optimizer(self,loss, learning_rate):
        with tf.name_scope("optimizer_model"):
            tvars = [v for v in tf.trainable_variables()]
            clipped_gradient, gradient_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
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
