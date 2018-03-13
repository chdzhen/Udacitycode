import tensorflow as tf
import numpy as np
import cv2


filenames = 'test.tfrecord'
# Generate dataset from TFRecord file.
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.batch(32)
#shuffle ，打乱dataset中的元素，有一个参数buffsize，表示打乱时使用的buffer的大小
dataset = dataset.shuffle(buffer_size=10000)
#repeat 将整个序列重复多少次
dataset = dataset.repeat(10)

# Make dataset iteratable.
iterator = dataset.make_one_shot_iterator()
next_example = iterator.get_next()

# get feature from serialized example
features = tf.parse_single_example(next_example,
        features={
            'a': tf.FixedLenFeature([], tf.float32),
            'b': tf.FixedLenFeature([2], tf.int64),
            'c': tf.FixedLenFeature([], tf.string)
        }
    )

a_out = features['a']
b_out = features['b']
c_raw_out = features['c']
c_out = tf.decode_raw(c_raw_out, tf.uint8)
c_out = tf.reshape(c_out, [400, 600,3])

print (a_out)
print (b_out)
print (c_out)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

tf.train.start_queue_runners(sess=sess)
a_val, b_val, c_val = sess.run([a_batch, b_batch, c_batch])
# print(a_val, b_val, c_val)
print ('first batch:')
print ('  a_val:',a_val)
print ('  b_val:',b_val)
print ('  c_val:',c_val)

c_val = np.array(c_val ,np.uint8)
cv2.namedWindow('lena',cv2.WINDOW_AUTOSIZE)
cv2.imshow("lena",c_val[0])
cv2.waitKey()

a_val, b_val, c_val = sess.run([a_batch, b_batch, c_batch])
print ('second batch:')
print ('  a_val:',a_val)
print ('  b_val:',b_val)
print ('  c_val:',c_val)
sess.close()