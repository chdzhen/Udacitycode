'''
use numpy to write sigmoid
'''

import numpy as np

def sigmoid(x):
    '''
    sigmoid function !!!
    '''
    return 1/(1+np.exp(-x))

inputs = np.array([0.7,-0.3])
weights = np.array([0.1,0.8])
bias = -0.1

'''
dot()函数在二维空间上的点乘与三位空间上的点乘是不一样的。
'''
output = sigmoid(np.dot(inputs,weights)+bias)

print('Outout:')
print(output)
