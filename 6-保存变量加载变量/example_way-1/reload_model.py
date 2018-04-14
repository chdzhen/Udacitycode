import infence
import tensorflow as tf 

saver = tf.train.Saver()

#启动图

#加载图
with tf.Session() as sess:
    saver.restore(sess,save_path)

    test_accuracy = sess.run(
        accuracy,
        feed_dict={
            features:mnist.test.images,labels:mnist.test.labels})

print('Test Accuacy:{}'.format(test_accuracy))