"""
read csv example!!!
"""

import tensorflow as tf

filename_queue = tf.train.string_input_producer(["tf_read.csv","tf_read2.csv"]) #读完tf_read.csv后接着读取tf_read2.csv
reader = tf.TextLineReader()
key,value = reader.read(filename_queue)

record_default = [[1.],[1.],[1.],[1.]]
col1,col2,col3,col4 = tf.decode_csv(value,record_defaults=record_default)

features = tf.stack([col1,col2,col3])

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    sess.run(local_init_op)
    
    #Start populating the filename queue
    cooder = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=cooder)
    try:
        for i in range(30):
            example,label = sess.run([features,col4])
            print(example)
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        cooder.request_stop()
        cooder.join(threads)
