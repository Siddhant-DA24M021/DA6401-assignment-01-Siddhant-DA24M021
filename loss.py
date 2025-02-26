import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.preds = None
        self.targets = None

    def __call__(self, preds, targets):
        self.preds = preds # batch_size x num_class
        self.targets = targets # batch_size x num_class

        # Targets are One-Hot Encoded
        losses = -np.sum(targets * np.log(self.preds + 1e-15), axis = 1)
        return np.mean(losses)
    
    def backward(self):
        grad = - self.targets / (self.preds + 1e-15)
        return grad # Derivative of loss function w.r.t. predictions (y-hat)


class SquaredErrorLoss:
    def __init__(self):
        self.preds = None
        self.targets = None

    def __call__(self, preds, targets):
        self.preds = preds # batch_size x num_class
        self.targets = targets # batch_size x num_class

        # Targets are One-Hot Encoded
        losses = np.sum((self.targets - self.preds)**2, axis = 1)
        return np.mean(losses)
    
    def backward(self):
        grad = -2 * (self.targets - self.preds)
        return grad # Derivative of loss function w.r.t. predictions (y-hat)