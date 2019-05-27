#coding=utf-8

import tensorflow as tf
from PIL import Image
import numpy as np
import argparse
from tensorflow.python.framework.errors_impl import NotFoundError
from trainForward import forward
import os
from ErrorClass import  TestException,TrainException
import random
from config import  args

class Train:

    def __init__(self,args):
     self.args=args
     self.train_images_list = os.listdir(args.PATH_TRAIN)
     self.test_images_list = os.listdir(args.PATH_TEST)
     self.train_len=len(os.listdir(args.PATH_TRAIN))
     self.test_len = len(os.listdir(args.PATH_TEST))

    def img2gray(self,path,filename):
        """
        描述：将图片转换为灰度图 并且返回图片的数组形式
        :param img:
        :return:
        """
        img_file = os.path.join(path, filename)
        img=np.array(Image.open(img_file))
        if img.shape[2]==3:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray.reshape(self.args.IMG_H,self.args.IMG_W,self.args.IMG_C)
        else:
            return img.reshape(self.args.IMG_H,self.args.IMG_W,self.args.IMG_C)

    def text2OneHot(self,text):
        """
        描述：将文本转换为onehot数组
        :param text:
        :return:
        """
        length=len(text)
        if len(text)!=self.args.MAX_CAPTCHA:
            return TrainException("验证码不符合长度要求")
        #   4*36
        vector = np.zeros(shape=self.args.MAX_CAPTCHA*self.args.CHAR_LEN)
        for i, item in enumerate(text):
            index=self.args.CHAR_LEN*i
            index=index+self.args.CHAR_SET.index(item)
            vector[index]=1
        return vector

    def getBatch(self,batch,is_train=True):
        """
            获取批量的训练数据
        :return:
        """
        #  128 * 100 * 60 * 1
        #batch_labels=[]
        batch_inputs=np.zeros(shape=[self.args.BATCH_SIZE,self.args.IMG_H,self.args.IMG_W,self.args.IMG_C],dtype=np.float32)
        batch_labels=np.zeros(shape=[self.args.BATCH_SIZE,self.args.CHAR_LEN*self.args.MAX_CAPTCHA],dtype=np.float32)
        #最大批次
        max_batch=int(self.args.STEPS/self.args.BATCH_SIZE)

        if max_batch<1:
            return TrainException("训练或测试图片太少，不够一个批次的！")
        if batch > max_batch-1:
          batch=batch%max_batch

        a_index=batch*self.args.BATCH_SIZE
        b_index=(batch+1)*self.args.BATCH_SIZE
        if is_train==True:
            if a_index > self.train_len:
                a_index = a_index - self.train_len
                b_index = b_index - self.train_len
        else:
            if a_index>self.test_len:
                a_index=a_index-self.test_len
                b_index=b_index-self.test_len
        thisBatch=[]
        if is_train==True:
            thisBatch=self.train_images_list[a_index:b_index]
        else:
            thisBatch==self.train_images_list[a_index:b_index]
        for i ,filename in enumerate(thisBatch):
            inputs=self.img2gray(self.args.PATH_TRAIN,filename)
            label = self.text2OneHot(filename.split('_')[1])
            batch_inputs[i, :, :, :] = inputs / 255
            batch_labels[i, :] = label
        return batch_labels, batch_inputs

    def getBatchTest(self):
        """
            获取批量的训练数据
        :return:  labels， inputs
        """
        batch_inputs = np.zeros(shape=[self.args.BATCH_SIZE, self.args.IMG_H, self.args.IMG_W, self.args.IMG_C],dtype=np.float32)
        # 128 * 144
        batch_labels = np.zeros(shape=[self.args.BATCH_SIZE, self.args.CHAR_LEN * self.args.MAX_CAPTCHA],dtype=np.float32)
        if len(self.test_images_list)<self.args.BATCH_SIZE:
            return  TestException("测试集的测试图片太少了")
        #   随机从数组中选取了100个
        verify_images = random.sample(self.test_images_list, self.args.BATCH_SIZE)
        for i, filename in enumerate(verify_images):
            inputs = self.img2gray(self.args.PATH_TEST, filename)
            label = self.text2OneHot(filename.split('_')[1])
            batch_inputs[i, :, :, :] = inputs/255
            batch_labels[i, :] = label
        return batch_labels, batch_inputs


    def train(self):
        inputs = tf.placeholder(tf.float32, [None, args.IMG_H, args.IMG_W, args.IMG_C])
        is_train = tf.placeholder(tf.bool)
        y_predict =forward(inputs,is_train)#预测值  4*36
        y_labels = tf.placeholder(tf.float32, [self.args.BATCH_SIZE, self.args.MAX_CAPTCHA * self.args.CHAR_LEN])
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=y_labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.args.LEARNING_RATE).minimize(cost)
        #[ 128 , 4 ,36]
        predict = tf.reshape(y_predict,[-1, self.args.MAX_CAPTCHA,self.args.CHAR_LEN])  # 预测结果
        max_idx_p = tf.argmax(predict, 2)  # 预测结果  128 * 4 * 36
        reallabel=tf.reshape(y_labels, [-1, self.args.MAX_CAPTCHA, self.args.CHAR_LEN])
        max_idx_l = tf.argmax(reallabel, 2)  # 标签
        # 计算准确率
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))
        # 模型保存对象
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            if os.path.exists(self.args.PATH_MODEL):
                try:
                    saver.restore(sess, self.args.PATH_MODEL)
                # 判断捕获model文件夹中没有模型文件的错误
                except NotFoundError:
                    print("model文件夹为空，将创建新模型")
            else:
                pass
            step = 1
            for i in range(self.args.STEPS):
                # 每个批次100张图片
                batch_label, batch_input = self.getBatch(i,is_train=True)
                # 梯度下降训练
                _, cost_ = sess.run([optimizer, cost],
                                    feed_dict={inputs:batch_input,is_train:True,y_labels:batch_label})
                if step % 10 == 0:
                    # 基于训练集的测试

                    batch_label_test, batch_input_test = batch_label, batch_input
                    acc_char,acc_img = sess.run([accuracy_char_count,accuracy_image_count],
                                        feed_dict={inputs: batch_input_test, is_train: False, y_labels: batch_label_test})
                    print("第{}批次 >>> ".format(i))
                    print("[训练集] 字符准确率为 {:.5f} 图片准确率为 {:.5f} >>> loss {:.10f}".format(acc_char, acc_img, cost_))

                    # 基于验证集的测试
                    batch_label_verify, batch_input_verify = self.getBatchTest()
                    acc_char,acc_img = sess.run([accuracy_char_count,accuracy_image_count],
                                        feed_dict={inputs: batch_input_verify,y_labels: batch_label_verify, is_train: False})
                    #print("第{}次训练 >>> ".format(step))
                    print("[验证集] 字符准确率为 {:.5f} 图片准确率为 {:.5f} >>> loss {:.10f}".format(acc_char, acc_img, cost_))
                    # 准确率达到99%后保存并停止
                    if acc_char > 0.99:
                        saver.save(sess, self.args.PATH_MODEL)
                        print("验证集准确率达到99%，保存模型成功")
                        break
                # 每训练500轮就保存一次
                if i % 500 == 0:
                    saver.save(sess, self.args.PATH_MODEL)
                    print("定时保存模型成功")
                step += 1
            saver.save(sess, self.args.PATH_MODEL)

train=Train(args)
train.train()


