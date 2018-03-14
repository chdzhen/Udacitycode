"""
read csv example!!!
"""

import tensorflow as tf

_CSV_COLUMN_DEFAULTS = [[1.],[1.],[1.],[1.]]
_CSV_COLUMNS = ['age','workclass','education','education_num']


# convert text to list of tensors for each column
def parseCSVLine(value):
    columns = tf.decode_csv(value,_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS,columns))
    return features

dataset = tf.data.TextLineDataset('tf_read.csv')

dataset = dataset.map(parseCSVLine)
iterator = dataset.make_one_shot_iterator()
textline = iterator.get_next()

with tf.Session() as sess:
    print(sess.run(textline))




