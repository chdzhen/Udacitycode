import infence
import math
import numpy as tf 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

save_file = './train_model.ckpt'
batch_size = 128
n_epochs = 100

saver = tf.train.Saver()

#启动图 如使用 infence.accuracy

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #训练循环
    for epoch in range(n_epochs):
        total_batch = math.ceil(mnist.train.num_examples/batch_size)

        #遍历所有 batch
        for i in range(total_batch):
            batch_featurs,batch_labels = mnist.train.next_batch(batch_size)
            sess.run(
                infence.optimizer,
                feed_dict={features:batch_featurs,labels:batch_labels})

        # 每运行10个epoch 打印一次状态
        if epoch % 10 == 0:
            vaild_accuracy = sess.run(
                accuracy,
                feed_dict={
                    features:mnist.validation.images,
                    labels:mnist.validation.labels})
            print('Epoch{:<3} - Validation Accuracy:{}'.format(epoch,vaild_accuracy))

#保存模型
saver.save(sess,save_file)
print('Trained Model Saved.')

     
