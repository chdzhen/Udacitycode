import tensorflow as tf

#The file path to save the data
save_path = './model/model.ckpt'

#两个Tensor变量：权重和偏置项
weights = tf.Variable(tf.truncated_normal([2,3]))
bias = tf.Variable(tf.truncated_normal([3]))

#用来保存Tensor变量的类
saver = tf.train.Saver()

with tf.Session() as sess:
    #初始化所有变量
    sess.run(tf.global_variables_initializer())

    #显示变量和权重
    print('Weights:')
    print(sess.run(weights))
    print('Bias:')
    print(sess.run(bias))

    #保存模型
    saver.save(sess,save_path)
