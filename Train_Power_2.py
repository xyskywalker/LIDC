# coding:utf-8
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.contrib import rnn
import pymysql
import os
from glob import glob


conn = pymysql.connect(host='localhost', user='root', passwd='111111', db='mydb', port=3306)


def load_data():
    strSQL = 'select ' \
            'record_date,  ' \
            'year(record_date) as year_1, ' \
            'case when year(record_date)=2015 then 1 else 2 end as year_2, ' \
            'quarter(record_date) as quarter_1, ' \
            'month(record_date) as month_1, ' \
            'day(record_date) as day_1, ' \
            'dayofweek(record_date) as dayofweek_1, ' \
            'total_power  ' \
            'from (SELECT record_date,sum(power_consumption) as total_power FROM mydb.Tianchi_power as a ' \
            'inner join ( ' \
            '  SELECT user_id FROM mydb.Tianchi_power_user_list where indexID<=1000 ' \
            '  union all ' \
            '  select user_id from (SELECT user_id FROM mydb.Tianchi_power_user_list where indexID>1000 order by RAND() ' \
            ' limit ' + str(np.random.randint(100,600)) + ') as b ' \
            '    union all ' \
            '    select -1 as user_id ' \
            ') as b ' \
            'on a.user_id = b.user_id ' \
            ' group by record_date order by record_date) as a'
    df = pd.read_sql(strSQL, conn)
    data = df.fillna(method='ffill')
    return data


def load_all_data():
    df = pd.read_sql('select ' +
                    'record_date, ' +
                    'year(record_date) as year_1,' +
                    'case when year(record_date)=2015 then 1 else 2 end as year_2,' +
                    'quarter(record_date) as quarter_1,' +
                    'month(record_date) as month_1,' +
                    'day(record_date) as day_1,' +
                    'dayofweek(record_date) as dayofweek_1,' +
                    'total_power ' +
                    'from (SELECT record_date,sum(power_consumption) as total_power FROM mydb.Tianchi_power' +
                    ' group by record_date order by record_date) as a', conn)

    data = df.fillna(method='ffill')
    return data

#data = load_data()# print(data)# data = np.load('output.npy')#print(data)#print(data.shape)#np.save('output.npy', data)
# print(data[:,1:8])
# print(np.mean(data[:,7:8], axis=0))
# print(data[:,2:8])# print(normalized_test_data)# print(normalized_test_data[:,5:6])
#normalized_train_data=(data[:,7:8]-np.mean(data[:,7:8],axis=0))/np.std(data[:,7:8],axis=0)
#print(mean)#print(std)#print(normalized_test_data)
# np.save('output.npy', data)
#定义常量
rnn_unit=30       # 隐藏单元数量
input_size=6 # 输入维度数量，6个：季度、月、日、星期、总用电量、上年同期用电总量(与输出日期一致)
output_size=1 # 输出维度数量，一个：用电量
lr=0.0006         #学习率
back_steps = 0

#——————————————————导入数据——————————————————————
# f=open('/Users/XY/PycharmProjects/datamining/tensorflow-program/rnn/stock_predict/dataset/dataset_2.csv')
# df=pd.read_csv(f)     #读入股票数据
# data=df.iloc[:,2:10].values  #取第3-10列
# f=open('/home/xy/output.csv')# df=pd.read_csv(f)# data=df.iloc[:,1:2].values
# print(data)# print(data.shape)

output_path = '/home/xy/PycharmRemote/tmpdata'
output_path_full = '/home/xy/PycharmRemote/tmpdata1/data_full.npy'

#for i in range(50000):#   print(i)#   data = load_data()#   np.save(os.path.join(output_path, "data_%06d.npy" % (i)), data)
#data = load_all_data()#np.save(os.path.join('/home/xy/PycharmRemote/tmpdata1', "data_full.npy"), data)

train_file_list = glob(output_path+"/data_??????.npy")

#获取训练集
def get_train_data(batch_size=60,time_step=20,train_begin=365,train_end=500, data=np.ndarray([])):
    batch_index=[]
    data_train=data[train_begin:train_end]

    normalized_train_data = np.zeros([train_end - train_begin, data.shape[1] - 1],dtype=np.float32)

    normalized_train_data[:, 0:6] = data_train[:, 2:8].astype(dtype=np.float32)

    normalized_train_data[:, 6:7] =  data[1:normalized_train_data.shape[0] + 1, 7:8]
    # 处理2016/2/29 这一天
    normalized_train_data[60:, 6:7] = data[59:normalized_train_data.shape[0] - 1 , 7:8]

    normalized_train_data = normalized_train_data[:, 1:7]

    mean = np.mean(normalized_train_data[:, 4:6], axis=0)
    std = np.std(normalized_train_data[:, 4:6], axis=0, dtype=np.float32)
    normalized_train_data[:, 4:6] = (normalized_train_data[:, 4:6] - mean) / std  # 标准化
    # print(normalized_train_data)    # print(normalized_train_data[:, 5:6])
    train_x,train_y=[],[]   #训练集
    for i in range(len(normalized_train_data)-time_step - back_steps - 1):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step + back_steps,:]
       y=normalized_train_data[i + 1 :i + time_step + back_steps + 1,4,np.newaxis]

       train_x.append(x.tolist())
       train_y.append(y.tolist())

    batch_index.append((len(normalized_train_data)-time_step))

    return batch_index,train_x,train_y,mean,std


#data_input = np.array(load_all_data())
#batch_index, train_x, train_y, mean_, std_ = get_train_data(60, 30, 365, 608-30,
#                                                            data=data_input)

#获取测试集
def get_test_data(batch_size=60, time_step=20,test_begin=581, last_value = -1.0, data=np.ndarray([])):
    batch_index = []

    data_test = np.array(data)
    data_test = data_test[test_begin: 639]

    normalized_test_data = np.zeros([639 - test_begin, data.shape[1] ], dtype=np.float32)

    normalized_test_data[:, 0:6] = data_test[:, 2:8].astype(dtype=np.float32)

    normalized_test_data[:, 6:7] =  data[2:normalized_test_data.shape[0]+2, 7:8]

    normalized_test_data = normalized_test_data[:, 1:7]

    #print(normalized_test_data[:,0:4])

    mean = np.mean(normalized_test_data[:, 4:6], axis=0)
    std = np.std(normalized_test_data[:, 4:6], axis=0, dtype=np.float32)
    normalized_test_data[:, 4:6] = (normalized_test_data[:, 4:6] - mean) / std  # 标准化
    # print(normalized_train_data)    # print(normalized_train_data[:, 5:6])
    test_x = normalized_test_data[ : time_step , :]

    test_y = normalized_test_data[time_step : time_step + time_step,4,np.newaxis]
    test_y_all = normalized_test_data[time_step : time_step + time_step , :]

    if last_value != -1:
        test_x[-1:,5] = last_value

    return mean,std,test_x,test_y,test_y_all


#data_test = np.array(load_all_data())

#mean, std, test_x, test_y, test_y_all = get_test_data(60, 30, 579 - 30, -1.0, data_test)

#data = np.load('output.npy')#batch_index_, test_x, test_y, mean_, std_ = get_train_data(batch_size=30,time_step=30,train_begin=0,train_end=579,data=data)#print(np.array(test_x).shape)#print(test_x[0])
#print(np.array(test_y).shape)#print(test_y[0])
# test_x_new = np.ndarray(shape=(30,11), dtype=np.float32)# test_x_new[0:30, 0:6] = test_x[0:30, 0:6]# test_x_new[0:30, 6:11] = test_x[0:30, 0:5]
# print(test_x_new.shape)# print(test_x_new[0])

# print(mean)# print(test_y.shape)
#——————————————————定义神经网络变量——————————————————#输入层、输出层权重、偏置
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
def train_lstm(batch_size=30,time_step=30,train_begin=0,train_end=608-30):
    X=tf.placeholder(tf.float32, shape=[None,time_step + back_steps ,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step + back_steps,output_size])

    pred,_=lstm(X)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint('/Users/XY/PycharmProjects/datamining/tensorflow-program/rnn/stock_predict/ckd')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #data_input = np.load(output_path_full)
        #batch_index, train_x, train_y, mean_, std_ = get_train_data(batch_size, time_step, train_begin, train_end,
        #                                                            data=data_input)
        #saver.restore(sess, module_file)
        for i in range(2000):
            data_input = np.load(train_file_list[i])
            batch_index, train_x, train_y, mean_, std_ = get_train_data(batch_size, time_step, train_begin, train_end,
                                                                        data=data_input)

            #print(batch_index)
            # #print(np.array(train_x).shape)
            # #print(np.array(train_y).shape)
            # #print(train_x)
            # #print(train_y)
            loss_ = 0.00
            for step in range(len(batch_index)-1):
                _,loss_t=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})
                loss_ += loss_t
            print(i, loss_ / (len(batch_index)-1))

            #if i % 200==0:
            #    print("保存模型：",saver.save(sess,'stock2.model',global_step=i))

        test_predict = np.zeros([20,back_steps + time_step]).tolist()

        last_ = -1.0
        data_test = np.load(output_path_full)

        for i_test in range(1):
            test_predict_ = np.zeros(back_steps).tolist()

            mean, std, test_x, test_y, test_y_all = get_test_data(batch_size, time_step, 579-30, last_, data_test)
            #print(test_x)
            #print(test_y)
            #print(test_y_all)

            for i_x in range(time_step):
                prob = sess.run(pred, feed_dict={X: [test_x]})
                predict = prob.reshape((-1))
                last_ = predict[-1:]
                test_predict_.extend(last_)

                test_x[0:-1] = test_x[1:]
                test_x[-1] = test_y_all[back_steps + i_x]
                test_x[-1][4] = last_
            test_predict[i_test] = test_predict_
            print(i_test)

        test_predict = np.mean(np.array(test_predict), axis=0)
        test_predict = test_predict.tolist()

        mean, std, test_x, test_y, test_y_all = get_test_data(batch_size, time_step, 579-30, -1.0, data_test)

        test_predict[0:back_steps] = test_y[0:back_steps]

        test_y = np.ravel(test_y)
        test_y = np.array(test_y) * std[0] + mean[0]
        test_predict = np.array(test_predict) * std[0] + mean[0]
        # acc = np.average(np.abs(test_predict[-30:] - test_y[-30:]) / test_y[-30:])  # 偏差
        print('--END--')
        # print('Acc lost: ', acc)
        test_predict = test_predict.astype(dtype=np.int32)
        test_y = test_y.astype(dtype=np.int32)

        i = 0
        for _predict in test_predict:
            print(_predict)
            i += 1
        # 以折线图表示结果        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y, color='r')
        plt.show()

train_lstm()


#————————————————预测模型————————————————————
def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    #Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    mean,std,test_x,test_y=get_test_data(time_step)
    pred,_=lstm(X)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复        #module_file = tf.train.latest_checkpoint()        #saver.restore(sess, module_file)
        test_predict=[]
        for step in range(len(test_x)-1):
          prob=sess.run(pred,feed_dict={X:[test_x[step]]})
          predict=prob.reshape((-1))
          test_predict.extend(predict)
        test_y=np.array(test_y)*std[7]+mean[7]
        test_predict=np.array(test_predict)*std[7]+mean[7]
        acc=np.average(np.abs(test_predict-test_y[:len(test_predict)])/test_y[:len(test_predict)])  #偏差        #以折线图表示结果        plt.figure()
        plt.plot(list(range(len(test_predict))), test_predict, color='b')
        plt.plot(list(range(len(test_y))), test_y,  color='r')
        plt.show()

# prediction()
