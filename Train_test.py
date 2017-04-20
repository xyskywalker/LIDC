import tensorflow as tf
import time
from datetime import datetime
import math
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import inception_v4
import inception_utils
slim = tf.contrib.slim


batch_size = 10
height, width = 299, 299
istart = 0
iend = 0
isize = batch_size


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


def randommask(img, size, num, centerX, centerY, radius):
    imgHeight = img.shape[0]
    imgWidth = img.shape[0]
    mask = np.ones([imgHeight, imgWidth])
    for i in range(num):
        minX, minY, maxX, maxY = makemask(size, centerX, centerY, imgWidth, imgHeight, radius)
        for x_i in range(minX, maxX):
            for y_i in range(minY, maxY):
                mask[y_i, x_i] = 0
    return img * mask

def makemask(size, centerX, centerY, imgWidth, imgHeight ,radius):
    startX = 0
    startY = 0
    maxX = imgWidth - size
    maxY = imgHeight - size
    startX = np.random.randint(low=0, high=maxX)
    startY = np.random.randint(low=0, high=maxY)
    node_min_x = int(centerX - radius)
    node_min_y = int(centerY - radius)
    node_max_x = int(centerX + radius)
    node_max_y = int(centerY + radius)

    # 判断节点是否在mask内部
    if node_min_x > startX and node_min_y > startY and node_max_x < (startX + size) and node_max_y < (startY + size):
        # 如果在的话递归调用再随机产生一个
        return makemask(size, centerX, centerY, imgWidth, imgHeight, radius)
    else:
        # 不在的话返回mask左上角和右下角坐标
        return startX, startY, startX+size, startY+size


def randomnodemask(img, size, num, centerX, centerY, radius):
    imgHeight = img.shape[0]
    imgWidth = img.shape[0]
    mask = np.ones([imgHeight, imgWidth])
    for i in range(num):
        minX, minY, maxX, maxY = makenodemask(size, centerX, centerY, imgWidth, imgHeight, radius)
        for x_i in range(minX, maxX):
            for y_i in range(minY, maxY):
                mask[y_i, x_i] = 0
    return img * mask


def makenodemask(size, centerX, centerY, imgWidth, imgHeight ,radius):
    startX = centerX - size - radius
    startY = centerY - size - radius
    maxX = centerX + size + radius
    maxY = centerY + size + radius
    startX = np.random.randint(low=startX, high=maxX)
    startY = np.random.randint(low=startY, high=maxY)

    return startX, startY, startX+size, startY+size


def cutimg(img, centerX, centerY):
    if centerX < 106:
        startX = int(centerX) - 20
    elif centerX > 405:
        startX = int(centerX) + 20 - 299
    else:
        startX = 106

    if centerY < 106:
        startY = int(centerY) - 20
    elif centerY > 405:
        startY = int(centerY) + 20 - 299
    else:
        startY = 106

    maxX = startX + 299
    maxY = startY + 299

    # 中心点太靠上
    if centerY - startY < 20:
        startY -= 20
        maxY -= 20
    # 中心点太靠左
    if centerX - startX < 20:
        startX -= 20
        maxX -= 20
    # 中心点太靠下
    if maxY - centerY < 20:
        startY += 20
        maxY += 20
    # 中心点太靠右
    if maxX - centerX < 20:
        startX += 20
        maxX += 20

    return img[startY:maxY, startX:maxX], startX, startY


# 输入
# 先在二维空间上测试，299*299，1个通道(单色图片)
x_ = tf.placeholder(dtype=tf.float32, shape=[None, height, width])
# label: 结节的中心坐标[, 0] = x [, 1] = y
y_ = tf.placeholder(dtype=tf.float32, shape=[None, 2])
inputs = tf.reshape(x_, (-1, height, width, 1))

inception_v4_arg_scope = inception_utils.inception_arg_scope

with slim.arg_scope(inception_v4_arg_scope()):
    logits, end_points = inception_v4.inception_v4(inputs, dropout_keep_prob=0.5)

# print_activations(logits)
# 输出层
W_out = weight_variable([1001, 2])
b_out = bias_variable([2])
# 输出([None, 2])
y_conv = tf.matmul(logits, W_out) + b_out
# 损失([])
loss = tf.sqrt(tf.reduce_mean(tf.square(y_ - y_conv)))
# 优化器
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

train_file_path = "traindata\\"
train_file_list = glob(train_file_path+"images_????_????.npy")

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)

for runs in range(200):
    train_x = np.array(np.zeros(shape=[isize, 299, 299], dtype=np.float32))
    train_y = np.array(np.zeros(shape=[isize, 2], dtype=np.float32))

    print("Runs " + str(runs + 1) + " start [" + str(datetime.now()) + "]")
    all_loss = 0.0
    for i in range(0, 100):
        istart = i*isize
        iend = istart + isize - 1
        iIndex = 0
        for i_ in range(istart, iend):
            imgfilename = train_file_list[i_]
            v_centerfilename = imgfilename.replace("images", "v_center")
            imgs = np.load(imgfilename)
            v_center = np.load(v_centerfilename)

            train_x[iIndex], startX, startY = cutimg(imgs[0], v_center[0], v_center[1])

            train_y[iIndex][0] = v_center[0] - startX
            train_y[iIndex][1] = v_center[1] - startY

            train_x[iIndex] = randommask(train_x[iIndex], 32, 5, train_y[iIndex][0], train_y[iIndex][1], 20)
            train_x[iIndex] = randomnodemask(train_x[iIndex], 3, 3, train_y[iIndex][0], train_y[iIndex][1], 10)


            # plt.imshow(train_x[iIndex])
            # plt.plot(int(train_y[iIndex][0]), int(train_y[iIndex][1]), 'ro', color='red')
            # plt.show()

            iIndex += 1

        _, loss_ = sess.run([optimizer, loss], feed_dict={x_: train_x, y_: train_y})
        all_loss += float(loss_)

        if (i + 1) % 10 == 0:
            print("Step ", str(i + 1), "Loss = ", str(all_loss/10.0))
            all_loss = 0

    if (runs + 1) % 10 == 0:
        saver.save(sess, "save/model_%04d.skpt" % runs)


# 测试
for i in range(20):
    test_imgfilename = train_file_list[1060+i]
    test_v_centerfilename = test_imgfilename.replace("images", "v_center")
    test_img = np.load(test_imgfilename)[1]
    test_v_center = np.load(test_v_centerfilename)
    test_x = np.array(np.zeros(shape=[1, 299, 299], dtype=np.float32))
    test_y = np.array(np.zeros(shape=[1, 2], dtype=np.float32))

    test_x[0], startX, startY = cutimg(test_img, test_v_center[0], test_v_center[1])
    test_y[0][0] = test_v_center[0] - startX
    test_y[0][1] = test_v_center[1] - startY

    y_e, loss_ = sess.run([y_conv, loss], feed_dict={x_: test_x, y_: test_y})

    print(test_y)
    print(y_e)
    print(loss_)

    plt.imshow(test_x[0])
    plt.plot(int(test_y[0][0]), int(test_y[0][1]), 'ro', color='yellow')
    plt.plot(int(y_e[0][0]), int(y_e[0][1]), 'ro', color='red')
    plt.show()