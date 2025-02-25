import numpy as np
from initializers import RandomUniformInitializer

class LinearLayer:
    def __init__(self, nin, nout, initializer = RandomUniformInitializer()):
        self.nin = nin
        self.nout = nout

        self.weights, self.bias = initializer.weightsandbiases(self.nin, self.nout)

        self.dweights = np.zeros_like(self.weights)
        self.dbias = np.zeros_like(self.bias)
        self.input = None

    def __call__(self, x):
        self.input = x
        return np.matmul(x, self.weights) + self.bias
    
    def backward(self, grad_output):
        self.dweights = np.matmul(self.input.T, grad_output)
        self.dbias = np.sum(grad_output, axis = 0 , keepdims=True)
        
        grad_input = np.matmul(grad_output, self.weights.T) # dL/dh
        return grad_input


    def parameters(self):
        return [self.weights, self.bias]
    
    def gradients(self):
        return [self.dweights, self.dbias]