#_*_ coding:utf-8 _*_
import tensorflow as tf
import numpy as np

class AlexNet(object):
    def __init__(self, X, keep_prob, num_classes = 1000,
                 train_layers=list(), weights_path='bvlc_alexnet.npy'):
        self.X = X
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.train_layers = train_layers
        self.weights_path = weights_path
        self.global_step = tf.Variable(1, trainable=False)

    def conv(self, X, filter_height, filter_width, num_filters,
             stride_y, stride_x, name, padding='SAME', groups=1):

        input_channels = int(X.get_shape()[-1])

        convolve = lambda x, w: tf.nn.conv2d(x, w,
                                             strides=[1, stride_y, stride_x, 1],
                                             padding=padding)

        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[filter_height, filter_width,
                                                      input_channels/groups, num_filters])
            biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(X,weights)
        else:
            input_groups = tf.split(value=X, num_or_size_splits=groups, axis=3)
            weight_groups = tf.split(value=weights, num_or_size_splits=groups, axis=3)
            output_groups = [convolve(x,w) for x,w in zip(input_groups,weight_groups)]
            conv = tf.concat(values=output_groups, axis=3)
        bias = tf.nn.bias_add(conv, biases)
        relu = tf.nn.relu(bias, name=scope.name)
        return relu

    def fc(self, X, num_out, name, relu=True):
        num_in = int(X.get_shape()[-1])
        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights', shape=[num_in, num_out])
            biases = tf.get_variable('biases', [num_out])
            bias = tf.nn.bias_add(tf.matmul(X, weights), biases)
        if relu:
            return tf.nn.relu(bias)
        else:
            return bias

    def max_pool(self, X, filter_height, filter_width,
                 stride_y, stride_x, name, padding='SAME'):
        return tf.nn.max_pool(X, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1],
                              padding=padding, name=name)

    def lrn(self, X, radius, alpha, beta, name, bias=1):
        return tf.nn.local_response_normalization(X, depth_radius=radius,
                                                  bias=bias, alpha=alpha,
                                                  beta=beta, name=name)

    def dropout(self, X, keep_prob, name):
        return  tf.nn.dropout(X, keep_prob, name=name)

    def create_model(self):
        #lst layer: conv -> lrn -> pool
        conv1 = self.conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        norm1 = self.lrn(conv1, 2, 1e-05, 0.75, name='norm1')
        pool1 = self.max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

        #2nd layer: conv -> lrn -> pool (split into 2 groups)
        conv2 = self.conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        norm2 = self.lrn(conv2, 2, 1e-05, 0.75, name='norm2')
        pool2 = self.max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

        #3rd layer: conv
        conv3 = self.conv(pool2, 3, 3, 384, 1, 1,name='conv3')

        #4th layer: conv (split into 2 groups)
        conv4 = self.conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        #5th layer: conv -> pool (split into 2 groups)
        conv5 = self.conv(conv4,3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = self.max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        #6th layer: fully-connected -> dropout
        dim = int(pool5.get_shape()[-1])*int(pool5.get_shape()[-2])*int(pool5.get_shape()[-3])
        flattened = tf.reshape(pool5, [-1, dim])
        fc6 = self.fc(flattened, 4096, name='fc6')
        dropout6 = self.dropout(fc6, self.keep_prob, name='dropout6')

        #7th layer: fully-connected -> dropout
        fc7 = self.fc(dropout6, 4096, name='fc7')
        dropout7 = self.dropout(fc7, self.keep_prob, name='dropout7')

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
        '''
        weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        '''
        weights_dict = np.load(self.weights_path, encoding='bytes').item()
        for op_name in weights_dict:
            if op_name not in self.train_layers:
                with tf.variable_scope(op_name, reuse=True):
                    for data in weights_dict[op_name]:
                        #Biases
                        if len(data.shape) == 1:
                            biases = tf.get_variable('biases', trainable=False)
                            sess.run(biases.assign(data))
                        #Weights
                        else:
                            weights = tf.get_variable('weights', trainable=False)
                            sess.run(weights.assign(data))

