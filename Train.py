# coding:utf-8

import numpy as np
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt


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
x_ = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512])
# label: 结节的中心坐标[, 0] = x [, 1] = y
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
# dropout参数
keep_prob = tf.placeholder(dtype=tf.float32)
# reshaped并加上高斯噪音
noise = tf.random_normal(shape=tf.shape(x_), mean=0.0, stddev=0.5, dtype=tf.float32)
x = tf.reshape(x_ + noise, [-1, 512, 512, 1])

# Layer 1
# 共享权重，尺寸：[1, 1, 1, 128]，1*1的卷积核尺寸、1个通道(黑白图像)、512个卷积核数量，即特征数量
W_conv1 = weight_variable([1, 1, 1, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)  # [None, 256, 256, 16]

# Layer 2
W_conv2 = weight_variable([1, 1, 64, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)  # [None, 128, 128, 8]

# Layer 3
W_conv3 = weight_variable([1, 1, 64, 32])
b_conv3 = bias_variable([32])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)  # [None, 64, 64, 4]

# Layer 4
W_conv4 = weight_variable([1, 1, 32, 32])
b_conv4 = bias_variable([32])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)  # [None, 32, 32, 1]

# Layer 5
W_conv5 = weight_variable([1, 1, 32, 16])
b_conv5 = bias_variable([16])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
h_pool5 = max_pool_2x2(h_conv5)  # [None, 16, 16, 1]

# Layer 6
W_conv6 = weight_variable([1, 1, 16, 16])
b_conv6 = bias_variable([1])
h_conv6 = tf.nn.relu(conv2d(h_pool5, W_conv6) + b_conv6)
h_pool6 = max_pool_2x2(h_conv6)  # [None, 8, 8, 1]

print_activations(h_pool1)
print_activations(h_pool2)
print_activations(h_pool3)
print_activations(h_pool4)
print_activations(h_pool5)
print_activations(h_pool6)


# 构建全连接层，设定为128个神经元
# 权重
W_fc1 = weight_variable([8*8*16, 512])
# 偏置
b_fc1 = bias_variable([512])
# 将卷积层的池化输出转换为一维
h_pool6_flat = tf.reshape(h_pool6, [-1, 8*8*16])
# 激活函数
h_fc1 = tf.nn.relu(tf.matmul(h_pool6_flat, W_fc1) + b_fc1)
# Dropout层，避免过拟合(输出尺寸:[None, 128])
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# 输出层
W_out = weight_variable([512, 2])
b_out = bias_variable([2])
# 输出([None, 2])
y_conv = tf.matmul(h_fc1_drop, W_out) + b_out
# 损失([])
loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))

# 优化器
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

#################################################################################

train_file_path = "traindata\\"
train_file_list = glob(train_file_path+"images_????_????.npy")

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    istart = 0
    iend = 0
    isize = 5

    for runs in range(50):
        train_x = np.array(np.zeros(shape=[isize*3, 512, 512], dtype=np.float32))
        train_y = np.array(np.zeros(shape=[isize*3, 2], dtype=np.float32))

        print("Runs " + str(runs) + " start")

        for i in range(0, 200):
            istart = i*isize
            iend = istart + isize - 1
            iIndex = 0
            avg_loss = 0.0
            for i_ in range(istart, iend):
                imgfilename = train_file_list[i_]
                v_centerfilename = imgfilename.replace("images", "v_center")
                imgs = np.load(imgfilename)
                v_center = np.load(v_centerfilename)

                train_x[iIndex] = imgs[0]
                train_x[iIndex+1] = imgs[1]
                train_x[iIndex+2] = imgs[2]
                train_y[iIndex][0] = train_y[iIndex + 1][0] = train_y[iIndex + 2][0] = v_center[0]
                train_y[iIndex][1] = train_y[iIndex + 1][1] = train_y[iIndex + 2][1] = v_center[1]

                iIndex += 1

            _, loss_ = sess.run([optimizer, loss], feed_dict={x_: train_x, y_: train_y, keep_prob: 0.5})

            if i % 10 == 0:
                print("Step ", str(i), "Loss = ", str(loss_))

    # 测试
    test_imgfilename = train_file_list[1063]
    test_v_centerfilename = test_imgfilename.replace("images", "v_center")
    test_img = np.load(test_imgfilename)[1]
    test_v_center = np.load(test_v_centerfilename)
    test_x = np.array(np.zeros(shape=[1, 512, 512], dtype=np.float32))
    test_y = np.array(np.zeros(shape=[1, 2], dtype=np.float32))
    test_x[0] = test_img
    test_y[0][0] = test_v_center[0]
    test_y[0][1] = test_v_center[1]

    y_e, loss_ = sess.run([y_conv, loss], feed_dict={x_: test_x, y_: test_y, keep_prob: 1.0})

    print(test_y)
    print(y_e)
    print(loss_)

    plt.imshow(test_x[0])
    plt.plot(int(test_y[0][0]), int(test_y[0][1]), 'ro', color='yellow')
    plt.plot(int(y_e[0][0]), int(y_e[0][1]), 'ro', color='red')
    plt.show()

    plt.imshow(test_x[0])
    plt.show()

    # print(train_y.shape)
    # print_activations(y_)
    # print_activations(y_conv)
    # print(sess.run(y_conv, feed_dict={x_: train_x, y_: train_y, keep_prob: 0.5}).shape)

