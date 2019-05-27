import argparse

parser = argparse.ArgumentParser()
# Batch大小，默认为128
parser.add_argument("--BATCH_SIZE", type=int, default=128)
# 输入图像尺寸
parser.add_argument("--IMG_W", type=int, default=100)
parser.add_argument("--IMG_H", type=int, default=60)
parser.add_argument("--IMG_C", type=int, default=1)

# 学习率
parser.add_argument("--LEARNING_RATE", type=float, default=0.1)
#训练图片保存路径
parser.add_argument("--PATH_TRAIN", type=str, default="../imgs/train/")
#测试图片保存路径
parser.add_argument("--PATH_TEST", type=str, default="../imgs/test/")
charset=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
parser.add_argument("--CHAR_SET", type=int, default=charset)
#b_alpha
#self.w_alpha = 0.01
#        self.b_alpha = 0.1
parser.add_argument("--W_ALPHA", type=int, default=0.01)
parser.add_argument("--B_ALPHA", type=int, default=0.1)

parser.add_argument("--TRAIN_KEEP_PROB", type=int, default=0.75)
parser.add_argument("--TEST_KEEP_PROB", type=int, default=1)
#学习率
parser.add_argument("--LEARNINGRATE", type=int, default=0.0001)
#验证码 字符个数
parser.add_argument("--MAX_CAPTCHA", type=int, default=4)
#字符集的长度
parser.add_argument("--CHAR_LEN", type=int, default=36)

# 生成模型路径
parser.add_argument("--PATH_MODEL", type=str, default="./model/")
# 训练轮数
parser.add_argument("--STEPS", type=int, default=15000)
args = parser.parse_args()