# -*- coding: UTF-8 -*-
"""
使用captcha lib生成验证码（前提：pip install captcha）
"""
from captcha.image import ImageCaptcha
import os
import random
import time
import argparse

parser = argparse.ArgumentParser()
# 输入图像尺寸
parser.add_argument("--IMGHIGHT", type=int, default=100)
parser.add_argument("--IMGWIDTH", type=int, default=60)
#字符集
parser.add_argument("--CHACTERS", type=str, default="0123456789abcdefghijklmnopqrstuvwxyz")
#生成图片的个数
parser.add_argument("--IMAGENUMS", type=int, default=20)
#图片保存路径
parser.add_argument("--ROOTDIR", type=str, default="../imgs/origin/")
#验证码包含的字符的个数
parser.add_argument("--CHARCOUNT", type=int, default=4)
# 图片格式
parser.add_argument("--IMGFORMAT", type=str, default="png")
args = parser.parse_args()


"""
    生成验证码图片
"""
class GenerateingDataset:

    def __init__(self,args):
        self.IMGHIGHT=args.IMGHIGHT
        self.IMGWIDTH = args.IMGWIDTH
        self.CHACTERS = args.CHACTERS
        self.IMAGENUMS = args.IMAGENUMS
        self.ROOTDIR = args.ROOTDIR
        self.IMGFORMAT = args.IMGFORMAT
        self.CHARCOUNT=args.CHARCOUNT

    def _genData(self,text,filename):
        generator = ImageCaptcha(width=self.IMGHIGHT, height=self.IMGWIDTH)  # 指定大小
        img = generator.generate_image(text)  # 生成图片
        img.save(filename)  # 保存图片


    def genDataSet(self):
        # 判断文件夹是否存在
        if not os.path.exists(self.ROOTDIR):
            os.mkdir(self.ROOTDIR)
        #遍历生成图片
        for i in range(self.IMAGENUMS):
            text = ""
            for j in range(self.CHARCOUNT):
                text += random.choice(self.CHACTERS)
            timec = str(time.time()).replace(".", "")
            p = os.path.join(self.ROOTDIR, "{}_{}.{}".format(text, timec, self.IMGFORMAT))
            self._genData(text, p)




ge=GenerateingDataset(args)
ge.genDataSet()

