'''
保存模型，保存变量，tf.add_to_collection('pred_network', y)
'''

import tensorflow as tf

#定义模型
input_x = tf.placeholder(tf.float32,shape=(None,in_dim),name='input_x')
input_y = tf.placeholder(tf.float32,shape=(None,out_dim),name='input_y')

w1 = tf.Variable(tf.truncated_normal([in_dim,h1_dim],stddev=0.1),name='w1')
b1 = tf.Variable(tf.zeros([h1_dim]),name='b1')
w2 = tf.Variable(tf.truncated_normal([h1_dim,out_dim]),name='w2')
b2 = tf.Variable(tf.zeros([out_dim]),name='b2')

keep_prob = tf.placeholder(tf.float32,name='keep_prob')

hidden1= tf.nn.relu(tf.matmul(input_x,w1)+b1)
hidden1_drop = tf.nn.dropout(hidden1,keep_prob)

#定义预测目标
y = tf.nn.softmax(tf.matmul(hidden1,w2)+b2)

#创建saver
saver = tf.train.Saver()

'''
tf.add_to_collection()
'''
#假如保存y,以便在预测时使用
tf.add_to_collection('pred_network',y)

with tf.Session() as sess:
    for step in range(1000):
        if step %100 ==0:
            #保存checkpoint,同时也默认导出一个meta_graph
            #graph 名为'my-model-{global_step}.meta'
            saver.save(sess,'my-model',global_step=step)