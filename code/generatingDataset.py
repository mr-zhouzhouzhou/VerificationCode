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
#训练图片的个数
parser.add_argument("--TRAINNUMS", type=int, default=19)
#测试图片的个数
parser.add_argument("--TESTNUMS", type=int, default=1)
#训练图片保存路径
parser.add_argument("--TRAINPATH", type=str, default="../imgs/train/")
#测试图片保存路径
parser.add_argument("--TESTPATH", type=str, default="../imgs/test/")
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
        self.args=args

    """
        生成图片    
    """
    def _genData(self,filepath):
        text = ""
        for j in range(self.args.CHARCOUNT):
            text += random.choice(self.args.CHACTERS)
        timec = str(time.time()).replace(".", "")
        filename = os.path.join(filepath, "{}_{}.{}".format(text, timec, self.args.IMGFORMAT))
        generator = ImageCaptcha(width=self.args.IMGHIGHT, height=self.args.IMGWIDTH)
        img = generator.generate_image(text)
        img.save(filename)
    """
        生成训练和测试的数据集
    """
    def genDataSet(self):
        # 判断文件夹是否存在
        if not os.path.exists(self.args.TRAINPATH):
            os.mkdir(self.args.TRAINPATH)
        if not os.path.exists(self.args.TESTPATH):
            os.mkdir(self.args.TESTPATH)
        #遍历生成图片
        for i in range(self.args.TRAINNUMS):
            self._genData(self.args.TRAINPATH)
        for i in range(self.args.TESTNUMS):
            self._genData(self.args.TESTPATH)




ge=GenerateingDataset(args)
ge.genDataSet()

