#coding=utf-8

import tensorflow as tf
from PIL import Image
import numpy as np
import scipy.misc as misc
import argparse
import time
from tensorflow.python.framework.errors_impl import NotFoundError
from forward import forward
import os
from ErrorClass import  TestException,TrainException

# 初始化各种参数
parser = argparse.ArgumentParser()
# 输入图像尺寸
parser.add_argument("--IMG_H", type=int, default=100)
parser.add_argument("--IMG_W", type=int, default=60)
parser.add_argument("--IMG_C", type=int, default=1)
#
parser.add_argument("--LABELS_NUMS", type=int, default=20000)
# Batch大小，默认为100
parser.add_argument("--BATCH_SIZE", type=int, default=100)
# 学习率
parser.add_argument("--LEARNING_RATE", type=float, default=0.001)
#训练图片保存路径
parser.add_argument("--PATH_TRAIN", type=str, default="../imgs/train/")
#测试图片保存路径
parser.add_argument("--PATH_TEST", type=str, default="../imgs/test/")
#训练图片的个数
parser.add_argument("--NUMS_TRAIN", type=int, default=19)
#测试图片的个数
parser.add_argument("--NUMS_TEST", type=int, default=1)

#验证码 字符个数
parser.add_argument("--max_captcha", type=int, default=4)
#字符集的长度
parser.add_argument("--char_set_len", type=int, default=26)

# 生成模型路径
parser.add_argument("--PATH_MODEL", type=str, default="./model/")
# 训练轮数
parser.add_argument("--steps", type=int, default=5000)
args = parser.parse_args()


def backward(args=args):
    pass


class Train:
    def __init__(self,args):
        self.args=args
        pass


    def _backward(self):
        pass


    #转换图片成一通道的
    def convert2gray(self,img):
        """
        图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
        :param img:
        :return:
        """
        if len(img.shape) > 2:
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        else:
            return img


    def text2vector(self, text):
        """
         将标签转换为oneHot编码
        :param text: str
        :return: numpy.array
        """
        text_len = len(text)
        if text_len > self.args.max_captcha:
            raise ValueError('验证码最长{}个字符'.format(self.args.max_captcha))

        vector = np.zeros(self.args.max_captcha * self.args.char_set_len)
        #  必须要设置成self.max_captcha * self.char_set_len的
        for i, ch in enumerate(text):
            idx = i * self.char_set_len + self.char_set.index(ch)
            vector[idx] = 1
        return vector



    def get_batch(self, n, size=100):
        batch_x = np.zeros([size, self.image_height * self.image_width])  # 初始化
        batch_y = np.zeros([size, self.max_captcha * self.char_set_len])  # 初始化
        max_batch = int(len(self.train_images_list) / size)#z最大批次
        # print(max_batch)
        if max_batch - 1 < 0:
            raise TrainException("训练集图片数量需要大于每批次训练的图片数量")
        if n > max_batch - 1:
            n = n % max_batch
        s = n * size
        e = (n + 1) * size
        this_batch = self.train_images_list[s:e]
        # print("{}:{}".format(s, e))

        for i, img_name in enumerate(this_batch):
            label, image_array = self.gen_captcha_text_image(self.train_img_path, img_name)
            image_array = self.convert2gray(image_array)  # 灰度化图片
            batch_x[i, :] = image_array.flatten() / 255  # flatten 转为一维
            batch_y[i, :] = self.text2vec(label)  # 生成 oneHot
        return batch_x, batch_y



    def train(self):
        # 内容图像：batch为100，图像大小为100*60*1
        inputs = tf.placeholder(tf.float32, [128, 100, 60, 1])
        y_predict=forward(inputs)
        # 交叉熵顺损失函数
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=self.Y))
        # 采用梯度下降
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
        # 计算准确率  4*26
        predict = tf.reshape(y_predict, [-1, self.max_captcha, self.char_set_len])  # 预测结果
        max_idx_p = tf.argmax(predict, 2)  # 预测结果
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, self.max_captcha, self.char_set_len]), 2)  # 标签
        # 计算准确率
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))
        # 模型保存对象
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            # 恢复模型
            if os.path.exists(self.model_save_dir):
                try:
                    saver.restore(sess, self.model_save_dir)
                # 判断捕获model文件夹中没有模型文件的错误
                except NotFoundError:
                    print("model文件夹为空，将创建新模型")
            else:
                pass
            step = 1
            for i in range(3000):
                batch_x, batch_y = self.get_batch(i, size=128)
                # 梯度下降训练
                _, cost_ = sess.run([optimizer, cost],
                                    feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                if step % 10 == 0:
                    # 基于训练集的测试
                    batch_x_test, batch_y_test = self.get_batch(i, size=100)
                    acc_char = sess.run(accuracy_char_count,
                                        feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    acc_image = sess.run(accuracy_image_count,
                                         feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    print("第{}次训练 >>> ".format(step))
                    print("[训练集] 字符准确率为 {:.5f} 图片准确率为 {:.5f} >>> loss {:.10f}".format(acc_char, acc_image, cost_))
                    # 基于验证集的测试
                    batch_x_verify, batch_y_verify = self.get_verify_batch(size=100)
                    acc_char = sess.run(accuracy_char_count,
                                        feed_dict={self.X: batch_x_verify, self.Y: batch_y_verify, self.keep_prob: 1.})
                    acc_image = sess.run(accuracy_image_count,
                                         feed_dict={self.X: batch_x_verify, self.Y: batch_y_verify, self.keep_prob: 1.})
                    print("[验证集] 字符准确率为 {:.5f} 图片准确率为 {:.5f} >>> loss {:.10f}".format(acc_char, acc_image, cost_))
                    # 准确率达到99%后保存并停止
                    if acc_image > 0.99:
                        saver.save(sess, self.model_save_dir)
                        print("验证集准确率达到99%，保存模型成功")
                        break
                # 每训练500轮就保存一次
                if i % 500 == 0:
                    saver.save(sess, self.model_save_dir)
                    print("定时保存模型成功")
                step += 1
            saver.save(sess, self.model_save_dir)






