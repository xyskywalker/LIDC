# coding:utf-8

import numpy as np
import scipy.misc
import tensorflow as tf
import SimpleITK as sitk
import csv
from glob import glob
import pandas as pd
import os
from tqdm import tqdm

luna_path = "C:\\Users\\XY\\LUNA16\\"
output_path = "output\\"
file_list = glob(luna_path+"subset0\\*.mhd")


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return f

df_node = pd.read_csv(luna_path+"CSVFILES\\annotations.csv")
df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
df_node = df_node.dropna()

'''
for fcount, img_file in enumerate(file_list):
    mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
    if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
        # load the data once
        itk_img = sitk.ReadImage(img_file)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        print(img_array.shape)
'''


# 显示网络每一层结构的函数，输出名称和尺寸
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


# 权重初始化函数
def weight_variable(shape):
    # 添加一些随机噪声来避免完全对称，使用截断正态分布，标准差为0.1
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial_value=initial)


# 偏置量初始化函数
def bias_variable(shape):
    # 为偏置量增加一个很小的正值(0.1)，避免死亡节点
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial_value=initial)


# 卷积函数
def conv2d(x, W):
    # x: 输入
    # W: 卷积参数，例如[5,5,1,32]：5,5代表卷积核尺寸、1代表通道数：黑白图像为1，彩色图像为3、32代表卷积核数量也就是要提取的特征数量
    # strides: 步长，都是1代表会扫描所有的点
    # padding: SAME会加上padding让卷积的输入和输出保持一致尺寸
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 最大池化函数
def max_pool_2x2(x):
    # 使用2*2进行最大池化，即把2*2的像素块降为1*1，保留原像素块中灰度最高的一个像素，即提取最显著特征
    # 横竖两个方向上以2为步长
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#################################################################################

# 输入
# 先在二维空间上测试，单张图片尺寸512*512，1个通道(单色图片)
x = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 1])
# label: 结节的中心坐标[, 0] = x [, 1] = y
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
# dropout参数
keep_prob = tf.placeholder(dtype=tf.float32)

# 第一层卷积
# 共享权重，尺寸：[3, 3, 1, 128]，3*3的卷积核尺寸、1个通道(黑白图像)、128个卷积核数量，即特征数量
W_conv1 = weight_variable([3, 3, 1, 128])
# 共享偏置量，128个=特征数量
b_conv1 = bias_variable([128])
# 激活函数
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
# 池化(输出尺寸：[None, 256, 256, 128])
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积
# 共享权重[5, 5, 128, 64]，5*5的卷积核尺寸、128个通道(第一个卷积层的输出特征数)、64个卷积核数量，即特征数量
W_conv2 = weight_variable([5, 5, 128, 64])
# 共享偏置量，64个=特征数量
b_conv2 = bias_variable([64])
# 激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# 池化(输出尺寸：[None, 128, 128, 64])
h_pool2 = max_pool_2x2(h_conv2)

# 第三层卷积
# 共享权重[3, 3, 64, 32]，3*3的卷积核尺寸、64个通道(第二个卷积层的输出特征数)、32个卷积核数量，即特征数量
W_conv3 = weight_variable([3, 3, 64, 32])
# 共享偏置量，32个=特征数量
b_conv3 = bias_variable([32])
# 激活函数
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# 池化(输出尺寸：[None, 64, 64, 32])
h_pool3 = max_pool_2x2(h_conv3)

# 第四层卷积
# 共享权重[3, 3, 32, 16]，3*3的卷积核尺寸、32个通道(第三个卷积层的输出特征数)、16个卷积核数量，即特征数量
W_conv4 = weight_variable([3, 3, 32, 16])
# 共享偏置量，16个=特征数量
b_conv4 = bias_variable([16])
# 激活函数
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
# 池化(输出尺寸：[None, 32, 32, 16])
h_pool4 = max_pool_2x2(h_conv4)

# 构建全连接层，设定为128个神经元
# 池化之后输出变成了 32*32 共有 16个特征，共计32*32*16
# 权重
W_fc1 = weight_variable([32*32*16, 128])
# 偏置
b_fc1 = bias_variable([128])
# 将卷积层的池化输出转换为一维
h_pool2_flat = tf.reshape(h_pool1, [-1, 32*32*16])
# 激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# Dropout层，避免过拟合(输出尺寸:[None, 128])
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 输出层
W_out = weight_variable([128, 2])
b_out = bias_variable([2])
# 输出([None, 2])
y_conv = tf.matmul(h_fc1_drop, W_out) + b_out
# 损失([])
loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))

#################################################################################

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print_activations(y_conv)
    print_activations(loss)






