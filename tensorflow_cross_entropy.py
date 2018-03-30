'''
TensorFlow 四种交叉熵函数:
1. tf.nn.sigmoid_cross_entropy_with_logits
2. tf.nn.softmax_cross_entropy_with_ligits
3. tf.nn.sparse_softmax_cross_entropy_with_logits
4. tf.nn.weighted_cross_entropy_with_logits

sigmoid_cross_entropy_with_logits:
    这个函数的输入是logits和targets，logits就是神经网络模型中的 W * X矩阵，
    注意不需要经过sigmoid，而targets的shape和logits相同，就是正确的label值，
    例如这个模型一次要判断100张图是否包含10种动物，这两个输入的shape都是[100, 10]。注释中还提到这10个分类之间是独立的、不要求是互斥，这种问题我们成为多目标，
    例如判断图片中是否包含10种动物，label值可以包含多个1或0个1，还有一种问题是多分类问题，
    例如我们对年龄特征分为5段，只允许5个值有且只有1个值为1，这种问题可以直接用这个函数吗？答案是不可以!!!

    意思就是：这个函数中lable中没有要求有只有一个类别（要求独立，不要求互斥！！！）label=[1,0,1,0,1] 
    logits: shape=[batch,classes],不需要sigmod
    label： shape=[batch,classes], 需要one-hot

weighted_sigmoid_cross_entropy_with_logits：
    weighted_sigmoid_cross_entropy_with_logits 是sigmoid_cross_entropy_with_logits的拓展版，输入参数和实现和后者差不多，可以多支持一个pos_weight参数，
    目的是可以增加或者减小正样本在算Cross Entropy时的Loss。实现原理很简单，在传统基于sigmoid的交叉熵算法上，正样本算出的值乘以某个系数接口。

softmax_cross_entropy_with_ligits: 
    但这里要求分类的结果是互斥的，保证只有一个字段有值 label=[1,0,0,0,0] 分类结果要求互斥！！！
    例如CIFAR-10中图片只能分一类而不像前面判断是否包含多类动物 
    logits与label的shape是一样的
    label需要进行one-hot
    logits不需要做sigmod也不需要做softmax(因为函数内部有更高效的softmax)


sparse_softmax_cross_entropy_with_logits:
    label的值必须是从0开始编码的int32或int64，而且值范围是[0, num_class)，步长为1.
    logits 不需要做sigmod也不需要做softmax
    label 不需要进行one-hot(函数内部有更高效的one-hot)
'''



