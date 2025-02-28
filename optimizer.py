import numpy as np


# Stochastic Gradient Descent
class SGD:
    def __init__(self, parameters = None, learning_rate = 0.001):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self, gradients):
        for parameter, grad in zip(self.parameters, gradients):
            parameter -= self.learning_rate * grad


# Momentum Based Gradient Descent
class MomentumGD:
    def __init__(self, parameters = None, learning_rate = 0.001, momentum = 0.9):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.u = [np.zeros_like(param) for param in parameters]

    def step(self, gradients):
        for i, (parameter, grad) in enumerate(zip(self.parameters, gradients)):
            self.u[i] = self.momentum * self.u[i] + self.learning_rate * grad
            parameter -= self.u[i]


# Nesterov Accelerated Gradient Descent
class NesterovAccGD:
    def __init__(self, parameters = None, learning_rate = 0.001, momentum = 0.9):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = [np.zeros_like(param) for param in parameters]

    def step(self, gradients):
        for i, (parameter, grad) in enumerate(zip(self.parameters, gradients)):
            self.u[i] = self.momentum * self.u[i] + self.learning_rate * grad
            parameter -= self.u[i]

# RMSProp
class RMSProp():
    def __init__(self, parameters = None, learning_rate = 0.01, beta = 0.5, epsilon = 1e-6):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.u = [np.zeros_like(param) for param in parameters]

    def step(self, gradients):
        for i, (parameter, grad) in enumerate(zip(self.parameters, gradients)):
            self.u[i] = self.beta * self.u[i] + (1 - self.beta) * grad**2
            effective_lr = self.learning_rate / (self.u[i] + self.epsilon)**0.5
            parameter -= effective_lr * grad

# Adam
class Adam():
    pass

# Nadam
class Nadam():
    pass