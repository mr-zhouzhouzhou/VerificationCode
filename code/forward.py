#coding=utf-8


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os
from tensorflow.python.framework.errors_impl import NotFoundError
from ErrorClass import TrainException,TestException


def forward(args,inputs):
    """
    :param self:
    :param inputs:
    :param weight:
    :return:
    """
    def _convModel(name,inputs,weights,strides):
        """
        :param inputs: 输入
        :param weights: 权重  3*3*1*32
        :return:
        """
        # 使用正态分布初始化核
        kernel = tf.get_variable(name , weights, dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        # 0初始化bias
        bias = tf.get_variable(name + "B", [weights[3]], initializer=tf.constant_initializer(0.))
        input = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], "SAME") + bias
        relu=tf.nn.relu(tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], "SAME") + bias)
        pool=tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        output=tf.nn.dropout(pool,0.3)
        return output


    #卷积1
    conv1 = _convModel("CONV1", inputs, [3, 3, 1, 32],2)
    #卷积2
    conv2 = _convModel("CONV2", conv1, [3, 3, 32, 64],2)
    #卷积3
    conv3 = _convModel("CONV3", conv2, [3, 3, 64, 128],2)


    next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]
    # 全连接层1
    wd1 = tf.get_variable(name='wd1', shape=[next_shape, 1024], dtype=tf.float32,
                          initializer=tf.contrib.layers.xavier_initializer())
    bd1 = tf.Variable(args.b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, wd1.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
    dense = tf.nn.dropout(dense, args.keep_prob)

    # 全连接层2  4*26
    wout = tf.get_variable('name', shape=[1024, args.max_captcha * args.char_set_len], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    bout = tf.Variable(args.b_alpha * tf.random_normal([args.max_captcha * args.char_set_len]))
    y_predict = tf.add(tf.matmul(dense, wout), bout)
    return y_predict


