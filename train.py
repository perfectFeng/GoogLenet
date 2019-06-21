#!/usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import random
import sys
import os
import event_read
import GoogleNet
import brnn_model
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_dir = "chckPts/"
save_prefix = "save"
summaryFolderName = "summary/"

model_filename1 = "./chckPts/inception_v1.ckpt"
model_filename2 = "./chckPts/save46045.ckpt"
start_step = 0

batch_size = GoogleNet.batch_size
num_frames = GoogleNet.num_frames
height = GoogleNet.height
width = GoogleNet.width
channels = GoogleNet.channels
n_classes = GoogleNet.NUM_CLASSES
hidden_size = brnn_model.hidden_size

max_iters = 8


def _variable_with_weight_decay(name, shape, wd):
    var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var


def calc_reward(logit):
    cross_entropy_mean = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_placeholder, logits=logit)
    )
    tf.summary.scalar(
        'cross_entropy',
        cross_entropy_mean
    )
    weight_decay_loss = tf.reduce_mean(tf.get_collection('weightdecay_losses'))

    tf.summary.scalar('weight_decay_loss', weight_decay_loss)
    total_loss = cross_entropy_mean + weight_decay_loss
    tf.summary.scalar('total_loss', total_loss)
    return total_loss


def tower_acc(logit, labels):
    correct_pred = tf.equal(tf.argmax(logit, 1), labels)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def evaluate():
    nextX, nextY = event_read.readTestFile(batch_size, num_frames)
    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}
    r = sess.run(accuracy, feed_dict=feed_dict)

    print("ACCURACY: " + str(r))


with tf.device('/gpu:1'):
    with tf.Graph().as_default():

        labels_placeholder = tf.placeholder(tf.int64, shape=batch_size)
        inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_frames, height, width, channels))

        with tf.variable_scope('b_rnn'):
            rw = _variable_with_weight_decay('rw', [hidden_size, n_classes], 0.0005)
            rb = _variable_with_weight_decay('rb', [n_classes], 0.000)

        feature = GoogleNet.inference_feature(inputs_placeholder)
        outputs = brnn_model.BiRNN(feature, rw, rb)

        loss = calc_reward(outputs)

        param = tf.trainable_variables()

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss, var_list=param)

        accuracy = tower_acc(outputs, labels_placeholder)
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()

        with tf.Session() as sess:
            saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            sess.run(init)
            variable1 = tf.contrib.framework.get_variables_to_restore(
                include=['InceptionV1/Mixed_3c/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4b/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4b/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4b/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_3c/Branch_1/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_3x3/weights',
                         'InceptionV1/Mixed_3c/Branch_3/Conv2d_0b_1x1/weights',
                         'InceptionV1/Mixed_4f/Branch_1/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4f/Branch_3/Conv2d_0b_1x1/weights',
                         'InceptionV1/Mixed_5b/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4c/Branch_1/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights',
                         'InceptionV1/Mixed_4f/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4f/Branch_2/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_5c/Branch_3/Conv2d_0b_1x1/weights',
                         'InceptionV1/Mixed_3c/Branch_2/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_5b/Branch_3/Conv2d_0b_1x1/weights',
                         'InceptionV1/Mixed_4d/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_5c/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4e/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4e/Branch_2/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4d/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4c/Branch_2/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4e/Branch_3/Conv2d_0b_1x1/weights',
                         'InceptionV1/Mixed_4f/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_5c/Branch_2/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4d/Branch_1/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4f/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_3b/Branch_3/Conv2d_0b_1x1/weights',
                         'InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Conv2d_2c_3x3/weights',
                         'InceptionV1/Mixed_4c/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_5b/Branch_1/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_5b/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_5c/Branch_1/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4d/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4e/Branch_1/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_5c/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_3c/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Conv2d_1a_7x7/weights',
                         'InceptionV1/Conv2d_2b_1x1/weights',
                         'InceptionV1/Mixed_4b/Branch_1/Conv2d_0b_3x3/weights',
                         # 'InceptionV1/Logits/Conv2d_0c_1x1/weights',
                         'InceptionV1/Mixed_4d/Branch_2/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_3b/Branch_2/Conv2d_0b_3x3/weights',
                         # 'InceptionV1/Logits/Conv2d_0c_1x1/biases',
                         'InceptionV1/Mixed_3c/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_5b/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4d/Branch_3/Conv2d_0b_1x1/weights',
                         'InceptionV1/Mixed_4e/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4c/Branch_1/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4b/Branch_2/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4e/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_3b/Branch_1/Conv2d_0b_3x3/weights',
                         'InceptionV1/Mixed_4c/Branch_0/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_5c/Branch_2/Conv2d_0a_1x1/weights',
                         'InceptionV1/Mixed_4c/Branch_3/Conv2d_0b_1x1/weights',
                         ])
            variable2 = tf.contrib.framework.get_variables_to_restore(
                include=['b_rnn/rw', 'b_rnn/rb'])
            restore1 = tf.train.Saver(variable1)
            restore2 = tf.train.Saver(variable2)

            restore1.restore(sess, model_filename1)
            restore2.restore(sess, model_filename2)

            summary_writer = tf.summary.FileWriter(summaryFolderName, graph=sess.graph)
            # training
            for epoch in range(max_iters):

                lines = event_read.readFile()

                for batch in range(int(len(lines) / batch_size)):

                    start_time = time.time()
                    nextX, nextY = event_read.readTrainData(batch, lines, batch_size, num_frames)

                    feed_dict = {inputs_placeholder: nextX, labels_placeholder: nextY}

                    _, summary, l, acc = sess.run([train_op, merged, loss, accuracy], feed_dict=feed_dict)

                    duration = time.time() - start_time

                    print('epoch-step %d-%d: %.3f sec' % (epoch, batch, duration))

                    if batch % 10 == 0:
                        saver.save(sess,
                                   save_dir + save_prefix + str(epoch * int(len(lines) / batch_size) + batch) + ".ckpt")
                        print('loss:', l, '---', 'acc:', acc)
                        summary_writer.add_summary(summary, epoch * int(len(lines) / batch_size) + batch)
                        evaluate()
