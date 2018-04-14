'''
Tensorflow 提供了 tf.nn.conv2d() 和 tf.nn.bias_add()函数创建自己的卷积层
tf.nn.max_pool() 用于卷积层实现最大池化
'''

import tensorflow as tf
import numpy as np

#Output depth
k_output = 64

# Image Properties
image_width = 10
image_height = 10
color_channels = 3

# Convolution filter
filter_size_width = 5
filter_size_height = 5

# Input/Image
input = tf.placeholder(
    tf.float32,
    shape=[None,image_height,image_width,color_channels])

# Weight and bias
weight = tf.Variable(
    tf.truncated_normal(
        [filter_size_height,filter_size_width,color_channels,k_output]))
bias = tf.Variable(tf.zeros(k_output))

# Apply Convolution
#strides 表示滤波器在input上移动的步长
conv_layer = tf.nn.conv2d(input,weight,strides=[1,2,2,1],padding='SAME')

#Add bias
conv_layer = tf.nn.bias_add(conv_layer,bias)

#Apply activation function
conv_layer = tf.nn.relu(conv_layer)

# Apply Max pooling
'''
ksize 和 strides 参数也被构建为四个元素的列表，
每个元素对应 input tensor 的一个维度 ([batch, height, width, channels])，
对 ksize 和 strides 来说，batch 和 channel 通常都设置成 1。
'''
conv_layer = tf.nn.max_pool(
    conv_layer,
    ksize=[1,2,2,1],
    strides=[1,2,2,1],
    padding='SAME')
