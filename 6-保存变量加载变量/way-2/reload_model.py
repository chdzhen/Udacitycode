'''
不需要重新定义网络结构的方法，加载模型变量
'''

import tensorflow as tf

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my_save_dir/my-model-1000.meta')
    new_saver.restore(sess,'my_save-dir/my-model-1000')

    #tf.get_collection() 返回一个list 但是这里只要第一个参数
    y = tf.get_collection('pred_network')[0]

    graph = tf.get_default_graph()

    # 因为y中有placeholder，所以sess.run(y)的时候还需要用实际待预测的样本以及相应的参数来填充这些placeholder，
    # 而这些需要通过graph的get_operation_by_name方法来获取。
    input_x = graph.get_operation_by_name('input_x').outputs[0]
    keep_drob = graph.get_operation_by_name('keep_prob').outputs[0]

    # 使用y进行预测
    sess.run(y,feed_dict={input_x:....,keep_drob:....})