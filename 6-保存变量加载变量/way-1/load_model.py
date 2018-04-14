import tensorflow as tf 

save_path='./model/model.ckpt'

#移除之前的权重
tf.reset_default_graph()

#两个变量：权重和偏置项
weights = tf.Variable(tf.truncated_normal([2,3]))
bias = tf.Variable(tf.truncated_normal([3]))

#用来保存Tensor变量
saver = tf.train.Saver()

with tf.Session() as sess:
    '''
    这里的不再需要初始化所有的变量，因为restore()函数中已经做了。
    '''
    #saver.restore(sess, 'results/model.ckpt.data-1000-00000-of-00001')
    #加载权重和偏置项
    saver.restore(sess,'./model/model.ckpt.data-00000-of-00001')
    print("Weight:")
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))
