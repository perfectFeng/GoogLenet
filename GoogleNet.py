#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf

# The UCF-101 dataset has 101 classes
NUM_CLASSES = 12

height = 224
width = 224
batch_size = 3
num_frames = 32
channels = 3


def conv2d(name, input, output_channels, kernel, s=None, biases=None, activation=True, batch_norm=True):
    with tf.variable_scope(name):
            if s is None:
                s = [1, 1, 1, 1]
            n_in = input.get_shape()[-1].value
            k = tf.get_variable(name='weights',
                                shape=[kernel[0], kernel[1], n_in, output_channels],
                                dtype=tf.float32,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                trainable=True)
            conv = tf.nn.conv2d(input, k, strides=s, padding='SAME', name=name)
            weight_decay = tf.nn.l2_loss(k) * 0.0005
            tf.add_to_collection('weightdecay_losses', weight_decay)
            if biases is not None:
                bias_init_val = tf.constant(0.0, shape=[biases], dtype=tf.float32)
                biases = tf.Variable(bias_init_val, trainable=True, name='b')
                conv = tf.nn.bias_add(conv, biases)
            if batch_norm:
                conv = tf.contrib.layers.batch_norm(conv, scope='batch_norm', is_training=True)
            if activation:
                conv = tf.nn.relu(conv)
    return conv


def max_pool(name, input, k, s):
    return tf.nn.max_pool(input, ksize=k, strides=s, padding='SAME', name=name)


def Inc(input, name, out_channel):
    with tf.variable_scope(name):
        with tf.variable_scope('Branch_0'):
            branch_0 = conv2d('Conv2d_0a_1x1', input, out_channel[0], [1, 1])

        with tf.variable_scope('Branch_1'):
            branch_1 = conv2d('Conv2d_0a_1x1', input, out_channel[1], [1, 1])
            branch_1 = conv2d('Conv2d_0b_3x3', branch_1, out_channel[2], [3, 3])

        with tf.variable_scope('Branch_2'):
            branch_2 = conv2d('Conv2d_0a_1x1', input, out_channel[3], [1, 1])
            branch_2 = conv2d('Conv2d_0b_3x3', branch_2, out_channel[4], [3, 3])

        with tf.variable_scope('Branch_3'):
            branch_3 = max_pool('MaxPool_0a_3x3', input, [1, 3, 3, 1], [1, 1, 1, 1])
            branch_3 = conv2d('Conv2d_0b_1x1', branch_3, out_channel[5], [1, 1])

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
    return net


def inference_feature(input):

    input = tf.reshape(input, [batch_size * num_frames, height, width, channels])

    with tf.variable_scope('InceptionV1'):
        # Conv2d_1a_7x7
        net = conv2d('Conv2d_1a_7x7', input, 64, [7, 7], [1, 2, 2, 1])

        # MaxPool_2a_3x3
        net = max_pool('MaxPool_2a_3x3', net, [1, 3, 3, 1], [1, 2, 2, 1])

        # Conv2d_2b_1x1
        net = conv2d('Conv2d_2b_1x1', net, 64, [1, 1])

        # Conv2d_2c_3x3
        net = conv2d('Conv2d_2c_3x3', net, 192, [3, 3])

        # MaxPool_3a_3x3
        net = max_pool('MaxPool3d_3a_3x3', net, [1, 3, 3, 1], [1, 2, 2, 1])

        # inception module: Mixed_3b
        net = Inc(net, 'Mixed_3b', [64, 96, 128, 16, 32, 32])
        # inception module: Mixed_3c
        net = Inc(net, 'Mixed_3c', [128, 128, 192, 32, 96, 64])

        # MaxPool3d_4a_3x3
        net = max_pool('MaxPool_4a_3x3', net, [1, 3, 3, 1], [1, 2, 2, 1])

        # inception module: Mixed_4b
        net = Inc(net, 'Mixed_4b', [192, 96, 208, 16, 48, 64])
        # inception module: Mixed_4c
        net = Inc(net, 'Mixed_4c', [160, 112, 224, 24, 64, 64])
        # inception module: Mixed_4d
        net = Inc(net, 'Mixed_4d', [128, 128, 256, 24, 64, 64])
        # inception module: Mixed_4e
        net = Inc(net, 'Mixed_4e', [112, 144, 288, 32, 64, 64])
        # inception module: Mixed_4f
        net = Inc(net, 'Mixed_4f', [256, 160, 320, 32, 128, 128])

        # MaxPool3d_5a_2x2
        net = max_pool('MaxPool_5a_2x2', net, [1, 2, 2, 1], [1, 2, 2, 1])

        # inception module: Mixed_5b
        with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
                branch_0 = conv2d('Conv2d_0a_1x1', net, 256, [1, 1])

            with tf.variable_scope('Branch_1'):
                branch_1 = conv2d('Conv3d_0a_1x1', net, 160, [1, 1])
                branch_1 = conv2d('Conv3d_0b_3x3', branch_1, 320, [3, 3])

            with tf.variable_scope('Branch_2'):
                branch_2 = conv2d('Conv3d_0a_1x1', net, 32, [1, 1])
                branch_2 = conv2d('Conv3d_0a_3x3', branch_2, 128, [3, 3])

            with tf.variable_scope('Branch_3'):
                branch_3 = max_pool('MaxPool_0a_3x3', net, [1, 3, 3, 1], [1, 1, 1, 1])
                branch_3 = conv2d('Conv3d_0b_1x1', branch_3, 128, [1, 1])

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)
        # inception module: Mixed_5c
        net = Inc(net, 'Mixed_5c', [384, 192, 384, 48, 128, 128])

        net = tf.nn.avg_pool(net, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')

    return net
