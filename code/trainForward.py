#coding=utf-8

import tensorflow as tf



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import random
import os
from tensorflow.python.framework.errors_impl import NotFoundError
from ErrorClass import TrainException,TestException
from config import  args


def forward(inputs,is_train=True):
    """
    :param self:
    :param inputs:
    :param weight:
    :return:
    """
    def _convModel(name,inputs,weights):
        """
        :param inputs: 输入
        :param weights: 权重  3*3*1*32  卷积核
        :return:
        """
        # 使用正态分布初始化核
        kernel = tf.get_variable(name , weights,dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.01))
        # 0初始化bias  输出
        bias =tf.Variable(args.B_ALPHA * tf.random_normal([weights[3]]))

        relu = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(inputs, kernel, strides=[1, 1, 1, 1], padding='SAME'), bias))
        pool=tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        if is_train==True:
            output = tf.nn.dropout(pool, args.TRAIN_KEEP_PROB)
        else:
            output = tf.nn.dropout(pool, args.TEST_KEEP_PROB)
        return output

    def _FC(name,inputs,weights=None,is_FC=True):
        """

        :param inputs: 输入
        :param weights: 权重的shape
        :return:   输出
        """
        w= tf.get_variable(name=name, shape=weights, dtype=tf.float32,
                           initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.Variable(args.B_ALPHA * tf.random_normal([weights[1]]))
        dense = tf.nn.relu(tf.add(tf.matmul(inputs, w), b))

        if is_train == True:
            output = tf.nn.dropout(dense, args.TRAIN_KEEP_PROB)
        else:
            output = tf.nn.dropout(dense, args.TEST_KEEP_PROB)

        return output



    #卷积1
    conv1 = _convModel("CONV1", inputs, [3, 3, 1, 32])
    #卷积2
    conv2 = _convModel("CONV2", conv1, [3, 3, 32, 64])
    #卷积3
    conv3 = _convModel("CONV3", conv2, [3, 3, 64, 128])

    #卷积到全链接的转换
    next_shape = conv3.shape[1] * conv3.shape[2] * conv3.shape[3]
    fcInput=tf.reshape(conv3,[-1,next_shape])

    # 全连接层1
    dense=_FC("FC1",fcInput,weights=[next_shape,1024])


    #   最后一层输出做 特殊处理
    wout = tf.get_variable('name', shape=[1024, args.MAX_CAPTCHA * args.CHAR_LEN], dtype=tf.float32,
                           initializer=tf.contrib.layers.xavier_initializer())
    bout = tf.Variable(args.B_ALPHA * tf.random_normal([args.MAX_CAPTCHA * args.CHAR_LEN]))
    y_predict = tf.add(tf.matmul(dense, wout), bout)
    print("y_predict :",y_predict[0])
    return y_predict


