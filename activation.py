# Activation functions
import numpy as np



# Softmax activation function
class Softmax:
    def __init__(self):
        self.output = None # Output Probabilities
    
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis = 1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis = 1, keepdims=True)
        return self.output
    
    def backward(self, grad_output):
        # derivative of loss function w.r.t. pre-activations going in the softmax layer
        # Would have been a single liner if the loss and output function would have been merged
        # But as mean squared loss is also asked, so I thought it would be better to make the softmax as standalone independent of loss function
        batch_size = self.output.shape[0]
        num_classes = self.output.shape[1]
        grad_input = np.zeros_like(grad_output)
        for i in range(batch_size):
            dy_dh = np.zeros((num_classes, num_classes))

            for j in range(num_classes):
                for k in range(num_classes):
                    if j == k:
                        dy_dh[j, k] = self.output[i, j] * (1 - self.output[i, j])
                    else:
                        dy_dh[j, k] = - self.output[i, j] * self.output[i, k]
            grad_input[i] = np.dot(grad_output[i], dy_dh)
        return grad_input

    def parameters(self):
        # This needs no parameters, this function is included because it is treated as a layer in my implementation as it should be for backprop
        return [] 
    
    def gradients(self):
        return []



# Sigmoid activation function
class Sigmoid:
    def __init__(self):
        self.output = None
    
    def __call__(self, x):
        # Clipping x values for numerical stability
        self.output = 1 / (1 + np.exp(-np.clip(x, -100, 100)))
        return self.output
    
    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)
    
    def parameters(self):
        # This needs no parameters, this function is included because it is treated as a layer in my implementation as it should be for backprop
        return [] 
    
    def gradients(self):
        return []



# Tanh  activation function
class Tanh:
    def __init__(self):
        self.output = None
    
    def __call__(self, x):
        # Using np.tanh for numerical stability
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        # Derivative of tanh(x) is 1 - tanh(x)^2
        return grad_output * (1 - self.output ** 2)
    
    def parameters(self):
        # This needs no parameters, this function is included because it is treated as a layer in my implementation as it should be for backprop
        return []
    
    def gradients(self):
        return []




# ReLU activation function
class ReLU:
    def __init__(self):
        self.output = None
        self.input = None
    
    def __call__(self, x):
        self.input = x
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, grad_output):
        # Derivative of ReLU is 1 for x > 0, 0 for x <= 0
        return grad_output * (self.input > 0)
    
    def parameters(self):
        # This needs no parameters, this function is included because it is treated as a layer in my implementation as it should be for backprop
        return []
    
    def gradients(self):
        return []
    
# Identity activation function
class Identity:
    def __init__(self):
        self.output = None
        self.input = None
    
    def __call__(self, x):
        self.input = x
        self.output = self.input
        return self.output
    
    def backward(self, grad_output):
        # Derivative of Identity is 1 
        return grad_output * 1
    
    def parameters(self):
        # This needs no parameters, this function is included because it is treated as a layer in my implementation as it should be for backprop
        return []
    
    def gradients(self):
        return []