'''
通过生成.pb文件保存变量，以及加载变量
'''

#保存计算图中的节点  
import tensorflow as tf  
from tensorflow.python.framework import graph_util  
  
v1 = tf.Variable(tf.constant(1.0,shape=[1]),name='v1')  
v2 = tf.Variable(tf.constant(2.0,shape=[1]),name='v2')  
res = tf.add(v1,v2,name='add_res')  
res2 = tf.add(res,v2,name='add_res1')  
  
with tf.Session() as sess:  
    sess.run(tf.global_variables_initializer())  
      
    #获取图  
    graph_def = tf.get_default_graph().as_graph_def()   
      
    #保存了两个节点。   
    #convert_variables_to_constants：通过这个函数可以将计算图中的变量及其取值通过常量保存。  
    #add_res 没有“:0”，表示这是计算节点，而“add_res:0” 表示节点计算后的输出张量。  
    output_graph_def = graph_util.convert_variables_to_constants(sess,graph_def,['add_res','add_res1'])       
      
    #文件句柄，保存文件。  
    with tf.gfile.GFile('./save/combined_model.pb','wb') as f:  
        f.write(output_graph_def.SerializeToString())  
          
#导入已保存的节点          
from tensorflow.python.platform import gfile  
with tf.Session() as sess:  
    model_filename = './save/combined_model.pb'          
    with gfile.FastGFile(model_filename,'rb') as f:  
        graph_def = tf.GraphDef()  
        graph_def.ParseFromString(f.read())  
      
    result1 = tf.import_graph_def(graph_def,return_elements=['add_res:0'])  #这里导入了之前保存的节点，并且“:0”表示该节点的第一次输出的结果    
    result2 = tf.import_graph_def(graph_def,return_elements=['add_res1:0'])   
    print(sess.run(result1))  
    print(sess.run(result2))  
