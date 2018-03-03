"""
Gradient descent one step 
"""

import numpy as np

def sigmod(x):
    """
    Calculate sigmod
    """
    return 1/(1+np.exp(-x))

def sigmod_prime(x):
    """
    Derivate of the sigmod of function
    """
    return sigmod(x)*(1-sigmod(x))

learnrate = 0.5
x = np.array([1,2,3,4])
y = np.array(0.5)

# Initial weights
w = np.array([0.5,-0.5,0.3,0.1])

# Calculate hedden
h = np.dot(x,w)

# Calculate output of neural network
nn_output = sigmod(h)

# Calculate error
# ture error is SSE ,this error is easy for Calculate
error = y- nn_output

# Calculate the error term
error_term = error*sigmod_prime(h)

# Calculate change in weights
del_w = learnrate * error_term *x


print('Neural Network output:')
print(nn_output)
print('Amount of error:')
print(error)
print('change in weights:')
print(del_w)