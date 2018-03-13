"""
dataset example!!!
"""
import tensorflow as tf
import numpy as np

#在 非Eager 模式下：
#----------------------------- first example -----------------------#
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))
iterator = dataset.make_one_shot_iterator()
one_element = iterator.get_next()
with tf.Session() as sess:
    for i in range(5):
        print(sess.run(one_element))
'''
with tf.Session() as sess:
    try:
        while True:
            print(sess.run(one_element))
    except tf.errors.OutOfRangeError:
        print("end!")
'''
#在 Eager 模式下
#----------------------------- second example ------------------------#
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execytion()
dataset = tf.data.Dataset.from_tensor_slices(np.array([1.0,2.0,3.0,4.0,5.0]))

for one_element in tfe.Iterator(dataset):
    print(one_element)#这里不用sess.run()



#-----------------------------------创建dataset----------------------------#
#切分第一个维度，会包含5个元素
dataset = tf.data.Dataset.from_tensor_slices(np.random.uniform(size=(5,2)))

#dict 创建 dataset
dataset = tf.data.Dataset.from_tensor_slices(
    {
        "a": np.array([1,2,3,4,5]),
        "b": np.random.uniform(size=(5,2))
    }
) 


# tuple 创建 dataset
dataset = tf.data.Dataset.from_tensor_slices(
    (
        np.array([1,2,3,4,5]),
        np.random.uniform(size=(5,2))
    )
)

#dataset转换map、batch、shuffle、repeat
#map：
dataset = tf.data.Dataset.from_tensor_slices(np.array([1,2,3,4,5]))
dataset = dataset.map(lambda x:x+1) #2,3,4,5,6

#batch,将dataset中的元素组成了大小为32的batch
dataset = dataset.batch(32)

#shuffle ，打乱dataset中的元素，有一个参数buffsize，表示打乱时使用的buffer的大小
dataset = dataset.shuffle(buffer_size=10000)

#repeat 将整个序列重复多少次
dataset = dataset.repeat(10)

#-----------------------------dataset example ----------------------#
# 函数的功能时将filename对应的图片文件读进来，并缩放到统一的大小
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string)
    image_resized = tf.image.resize_images(image_decoded, [28, 28])
    return image_resized, label

# 图片文件的列表
filenames = tf.constant(["/var/data/image1.jpg", "/var/data/image2.jpg", ...])
# label[i]就是图片filenames[i]的label
labels = tf.constant([0, 37, ...])

# 此时dataset中的一个元素是(filename, label)
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

# 此时dataset中的一个元素是(image_resized, label)
dataset = dataset.map(_parse_function)

# 此时dataset中的一个元素是(image_resized_batch, label_batch)
dataset = dataset.shuffle(buffersize=1000).batch(32).repeat(10)
