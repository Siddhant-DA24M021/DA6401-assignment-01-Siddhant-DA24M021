import numpy as np

class RandomUniformInitializer:
    def __init__(self, lower = -1, upper = 1):
        self.lower = lower
        self.upper = upper

    def weightsandbiases(self, nin, nout):
        weights = np.random.uniform(self.lower, self.upper, (nin, nout))
        biases = np.random.uniform(self.lower, self.upper, (1, nout))
        return weights, biases

    
class RandomNormalInitializer:
    def __init__(self, mu = 0.0, sigma = 1.0):
        self.mu = mu
        self.sigma = sigma

    def weightsandbiases(self, nin, nout):
        weights = np.random.normal(self.mu, self.sigma, (nin, nout))
        biases = np.random.normal(self.mu, self.sigma, (1, nout))
        return weights, biases
    
    
class XavierInitializer:
    def weightsandbiases(self, nin, nout):
        print("Hi")
        x = np.sqrt(6 / (nin + nout))
        weights = np.random.uniform(-x, x, (nin, nout))
        biases = np.random.uniform(-x, x, (1, nout))
        return weights, biases
