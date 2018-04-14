'''
构建模型
'''

import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

#移除之前的Tensor 和 运算
tf.reset_default_graph()

learning_rate = 0.001
n_input = 784
n_class = 10

#加载MNIST 数据
mnist = input_data.read_data_sets('.',one_hot=True)

#特征和标签
features = tf.placeholder(tf.float32,[None,n_input])
labels = tf.placeholder(tf.float32,[None,n_class])

weights = tf.Variable(tf.random_normal([n_input,n_class]))
bias = tf.Variable(tf.random_normal([n_class]))

#logits
logits = tf.add(tf.matmul(features,weights),bias)

#定义损失函数和优化器
cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

#计算准确率
correct_prediction = tf.equal(tf.arg_max(logits,1),tf.arg_max(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
