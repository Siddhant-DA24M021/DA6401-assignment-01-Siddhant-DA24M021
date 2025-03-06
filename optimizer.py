import numpy as np


# Stochastic Gradient Descent
class SGD:
    def __init__(self, parameters = None, learning_rate = 0.001, weight_decay=0.0001):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, gradients):
        for parameter, grad in zip(self.parameters, gradients):
            grad_with_decay = grad + self.weight_decay * parameter
            parameter -= self.learning_rate * grad_with_decay


# Momentum Based Gradient Descent
class MomentumGD:
    def __init__(self, parameters = None, learning_rate = 0.001, momentum = 0.9, weight_decay=0.0001):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.u = [np.zeros_like(param) for param in parameters]

    def step(self, gradients):
        for i, (parameter, grad) in enumerate(zip(self.parameters, gradients)):
            grad_with_decay = grad + self.weight_decay * parameter
            self.u[i] = self.momentum * self.u[i] + self.learning_rate * grad_with_decay
            parameter -= self.u[i]


# Nesterov Accelerated Gradient Descent
class NesterovAccGD:
    def __init__(self, parameters = None, learning_rate = 0.001, momentum = 0.9, weight_decay=0.0001):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.u = [np.zeros_like(param) for param in parameters]
        

    def apply_lookahead(self):
        for i, parameter in enumerate(self.parameters):
            parameter -= self.learning_rate * self.momentum * self.u[i]

    def step(self, gradients):
        for i, (parameter, grad) in enumerate(zip(self.parameters, gradients)):
            parameter += self.learning_rate * self.momentum * self.u[i] # restore original parameters (Undo lookahead)
            grad_with_decay = grad + self.weight_decay * parameter
            self.u[i] = self.momentum * self.u[i] + grad_with_decay
            parameter -= self.learning_rate * self.u[i]

# RMSProp
class RMSProp():
    def __init__(self, parameters = None, learning_rate = 0.01, beta = 0.5, epsilon = 1e-6, weight_decay=0.0001):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.u = [np.zeros_like(param) for param in parameters]

    def step(self, gradients):
        for i, (parameter, grad) in enumerate(zip(self.parameters, gradients)):
            grad_with_decay = grad + self.weight_decay * parameter
            self.u[i] = self.beta * self.u[i] + (1 - self.beta) * grad_with_decay**2
            effective_lr = self.learning_rate / np.sqrt(self.u[i] + self.epsilon)
            parameter -= effective_lr * grad_with_decay

# Adam
class Adam():
    def __init__(self, parameters = None, learning_rate = 0.001, beta1 = 0.5, beta2 = 0.5, epsilon = 1e-6, weight_decay=0.0001):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.t = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]

    def step(self, gradients):
        for i, (parameter, grad) in enumerate(zip(self.parameters, gradients)):
            grad_with_decay = grad + self.weight_decay * parameter
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_with_decay
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad_with_decay**2
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            effective_lr = self.learning_rate / (np.sqrt(v_hat) + self.epsilon)
            parameter -= effective_lr * m_hat
        self.t += 1


# Nadam
class Nadam():
    def __init__(self, parameters = None, learning_rate = 0.001, beta1 = 0.5, beta2 = 0.5, epsilon = 1e-6, weight_decay=0.0001):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.t = 1
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]

    def step(self, gradients):
        for i, (parameter, grad) in enumerate(zip(self.parameters, gradients)):
            grad_with_decay = grad + self.weight_decay * parameter
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_with_decay
            m_hat = self.m[i] / (1 - self.beta1 ** (self.t+1))
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad_with_decay**2
            v_hat = self.v[i] / (1 - self.beta2 ** (self.t+1))
            effective_lr = self.learning_rate / (np.sqrt(v_hat) + self.epsilon)
            parameter -= effective_lr * (self.beta1 * m_hat + (1 - self.beta1) * grad_with_decay / (1 - self.beta1 ** (self.t+1)))
        self.t += 1
