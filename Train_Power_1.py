#coding=utf-8

import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import pymysql
import os


conn = pymysql.connect(host='localhost', user='root', passwd='111111', db='mydb', port=3306)

def load_data():
    df = pd.read_sql('select * from (SELECT record_date,sum(power_consumption) FROM mydb.Tianchi_power as a ' +
    ' inner join (SELECT user_id FROM mydb.Tianchi_power_user_list order by RAND() limit ' + str(np.random.randint(1000,1454)) + ') as b ' +
    ' on a.user_id = b.user_id ' +
    ' group by record_date order by record_date) as a ' +
    ' union select \'2016-09-01 00:00:00\',2000000 ' +
    ' union select \'2016-09-02 00:00:00\',2000000 ' +
    ' union select \'2016-09-03 00:00:00\',2000000 ' +
    ' union select \'2016-09-04 00:00:00\',2000000 ' +
    ' union select \'2016-09-05 00:00:00\',2000000 ' +
    ' union select \'2016-09-06 00:00:00\',2000000 ' +
    ' union select \'2016-09-07 00:00:00\',2000000 ' +
    ' union select \'2016-09-08 00:00:00\',2000000 ' +
    ' union select \'2016-09-09 00:00:00\',2000000 ' +
    ' union select \'2016-09-10 00:00:00\',2000000 ' +
    ' union select \'2016-09-11 00:00:00\',2000000 ' +
    ' union select \'2016-09-12 00:00:00\',2000000 ' +
    ' union select \'2016-09-13 00:00:00\',2000000 ' +
    ' union select \'2016-09-14 00:00:00\',2000000 ' +
    ' union select \'2016-09-15 00:00:00\',2000000 ' +
    ' union select \'2016-09-16 00:00:00\',2000000 ' +
    ' union select \'2016-09-17 00:00:00\',2000000 ' +
    ' union select \'2016-09-18 00:00:00\',2000000 ' +
    ' union select \'2016-09-19 00:00:00\',2000000 ' +
    ' union select \'2016-09-20 00:00:00\',2000000 ' +
    ' union select \'2016-09-21 00:00:00\',2000000 ' +
    ' union select \'2016-09-22 00:00:00\',2000000 ' +
    ' union select \'2016-09-23 00:00:00\',2000000 ' +
    ' union select \'2016-09-24 00:00:00\',2000000 ' +
    ' union select \'2016-09-25 00:00:00\',2000000 ' +
    ' union select \'2016-09-26 00:00:00\',2000000 ' +
    ' union select \'2016-09-27 00:00:00\',2000000 ' +
    ' union select \'2016-09-28 00:00:00\',2000000 ' +
    ' union select \'2016-09-29 00:00:00\',2000000 ' +
    ' union select \'2016-09-30 00:00:00\',2000000'
        , conn)

    data = df.iloc[:,1:2].values

    return data

#定义常量
rnn_unit=20       #hidden layer units
input_size=1 #7
output_size=1
lr=0.0006         #学习率
#——————————————————导入数据——————————————————————
#f=open('/Users/XY/PycharmProjects/datamining/tensorflow-program/rnn/stock_predict/dataset/dataset_2.csv')
#df=pd.read_csv(f)     #读入股票数据
#data=df.iloc[:,2:10].values  #取第3-10列

# f=open('/home/xy/output.csv')
# df=pd.read_csv(f)
# data=df.iloc[:,1:2].values

# print(data)
# print(data.shape)

output_path = '/home/xy/PycharmRemote/tmpdata'

for i in range(10000):
    print(i)
    data = load_data()
    np.save(os.path.join(output_path, "data_%04d.npy" % (i)), data)


#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=0,train_end=500):
    batch_index=[]
    data_train=data[train_begin:train_end]

    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化


    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,:1]
       y=normalized_train_data[i+time_step :i+time_step+time_step,0,np.newaxis]

       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y



#获取测试集
def get_test_data(time_step=20,test_begin=581):
    df = pd.read_sql('select * from (SELECT record_date,sum(power_consumption) FROM mydb.Tianchi_power as a ' +
                     ' group by record_date order by record_date) as a ' +
                     ' union select \'2016-09-01 00:00:00\',2000000 ' +
                     ' union select \'2016-09-02 00:00:00\',2000000 ' +
                     ' union select \'2016-09-03 00:00:00\',2000000 ' +
                     ' union select \'2016-09-04 00:00:00\',2000000 ' +
                     ' union select \'2016-09-05 00:00:00\',2000000 ' +
                     ' union select \'2016-09-06 00:00:00\',2000000 ' +
                     ' union select \'2016-09-07 00:00:00\',2000000 ' +
                     ' union select \'2016-09-08 00:00:00\',2000000 ' +
                     ' union select \'2016-09-09 00:00:00\',2000000 ' +
                     ' union select \'2016-09-10 00:00:00\',2000000 ' +
                     ' union select \'2016-09-11 00:00:00\',2000000 ' +
                     ' union select \'2016-09-12 00:00:00\',2000000 ' +
                     ' union select \'2016-09-13 00:00:00\',2000000 ' +
                     ' union select \'2016-09-14 00:00:00\',2000000 ' +
                     ' union select \'2016-09-15 00:00:00\',2000000 ' +
                     ' union select \'2016-09-16 00:00:00\',2000000 ' +
                     ' union select \'2016-09-17 00:00:00\',2000000 ' +
                     ' union select \'2016-09-18 00:00:00\',2000000 ' +
                     ' union select \'2016-09-19 00:00:00\',2000000 ' +
                     ' union select \'2016-09-20 00:00:00\',2000000 ' +
                     ' union select \'2016-09-21 00:00:00\',2000000 ' +
                     ' union select \'2016-09-22 00:00:00\',2000000 ' +
                     ' union select \'2016-09-23 00:00:00\',2000000 ' +
                     ' union select \'2016-09-24 00:00:00\',2000000 ' +
                     ' union select \'2016-09-25 00:00:00\',2000000 ' +
                     ' union select \'2016-09-26 00:00:00\',2000000 ' +
                     ' union select \'2016-09-27 00:00:00\',2000000 ' +
                     ' union select \'2016-09-28 00:00:00\',2000000 ' +
                     ' union select \'2016-09-29 00:00:00\',2000000 ' +
                     ' union select \'2016-09-30 00:00:00\',2000000'
                     , conn)

    data = df.iloc[:, 1:2].values

    data_test=data[test_begin:]
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data= (data_test-mean)/std  #标准化
    size=(len(normalized_test_data)+time_step-1)//time_step  #有size个sample


    test_x,test_y=[],[]
    i = 0
    for i in range(size-1):
       x=normalized_test_data[i*time_step:(i+1)*time_step,:1]
       y=normalized_test_data[i*time_step+time_step:(i+1)*time_step+time_step,0]
       test_x.append(x.tolist())
       test_y.extend(y)
    test_x.append((normalized_test_data[(i+1)*time_step:,:1]).tolist())
    test_y.extend((normalized_test_data[(i+1)*time_step:,0]).tolist())



    return mean,std,test_x,test_y


# print(mean)
# print(test_y.shape)

#——————————————————定义神经网络变量——————————————————
#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=rnn.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#——————————————————训练模型——————————————————
def train_lstm(batch_size=90,time_step=30,train_begin=0,train_end=608):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])

    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint('/Users/XY/PycharmProjects/datamining/tensorflow-program/rnn/stock_predict/ckd')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        #重复训练10000次
        for i in range(3000):
            data = load_data()
            batch_index, train_x, train_y = get_train_data(batch_size, time_step, train_begin, train_end)

            for step in range(len(batch_index)-1):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
            print(i,loss_)
            #if i % 200==0:
            #    print("保存模型：",saver.save(sess,'stock2.model',global_step=i))


        mean, std, test_x, test_y = get_test_data(time_step, 579)

        test_predict = []
        for step in range(len(test_x) - 1):
            print('test step:' , step)
            print(np.array(test_x[step]) * std[0] + mean[0])
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            print('Result')
            print(np.array(predict) * std[0] + mean[0])
            test_predict.extend(predict)
        test_y = np.array(test_y) * std[0] + mean[0]
        test_predict = np.array(test_predict) * std[0] + mean[0]
        acc = np.average(np.abs(test_predict - test_y[:len(test_predict)]) / test_y[:len(test_predict)])  # 偏差

        print('--END--')
        print('Acc lost: ', acc)
        print(test_predict)
        # 以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()


# train_lstm()


#————————————————预测模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)

    pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        #module_file = tf.train.latest_checkpoint()
        #saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差
        #以折线图表示结果
        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

# prediction()