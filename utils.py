import numpy as np
from activation import Softmax, Sigmoid, Tanh, ReLU, Identity
from initializers import RandomUniformInitializer, RandomNormalInitializer, XavierInitializer
from loss import CrossEntropyLoss, SquaredErrorLoss
from optimizer import SGD, MomentumGD, NesterovAccGD, RMSProp, Adam, Nadam


# Train Validation split functionality for dataset
def train_val_split(X, y, val_ratio = 0.1):
    num_samples = X.shape[0]
    val_size = int(num_samples * val_ratio)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    X_train, X_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return X_train, y_train, X_val, y_val




def get_activation_function(name):
    activation_functions = {
        "softmax": Softmax,
        "identity": Identity,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "ReLU": ReLU,
    }
    if name not in activation_functions:
        raise ValueError(f"Invalid activation function name: {name}")
    return activation_functions[name]()




def get_initializer(name):
    initializers = {
        "random": RandomNormalInitializer,
        "uniform": RandomUniformInitializer,
        "Xavier": XavierInitializer
    }

    if name not in initializers:
        raise ValueError(f"Invalid initializer name: {name}")
    return initializers[name]()




def get_loss_function(name):
    loss_functions = {
        "mean_squared_error": SquaredErrorLoss, 
        "cross_entropy": CrossEntropyLoss
    }

    if name not in loss_functions:
        raise ValueError(f"Invalid loss function name: {name}")
    return loss_functions[name]()


def get_optimizer(name, model_params, **kwargs):

    optimizer = None
    learning_rate =  kwargs.pop("learning_rate", 0.001)

    if name == "sgd":
        optimizer = SGD(parameters = model_params, 
                        learning_rate = learning_rate)
        
    elif name == "momentum":
        momentum = kwargs.pop("momentum", 0.5)
        optimizer = MomentumGD(parameters = model_params, 
                               learning_rate = learning_rate, 
                               momentum = momentum)
        
    elif name == "nag":
        momentum = kwargs.pop("momentum", 0.5)
        optimizer = NesterovAccGD(parameters = model_params, 
                                  learning_rate = learning_rate, 
                                  momentum = momentum)
        
    elif name == "rmsprop":
        beta = kwargs.pop("beta", 0.5)
        optimizer = RMSProp(parameters = model_params, 
                            learning_rate = learning_rate,
                            beta = beta)
        
    elif name == "adam":
        beta1 = kwargs.pop("beta1", 0.5)
        beta2 = kwargs.pop("beta2", 0.5)
        optimizer = Adam(parameters = model_params, 
                            learning_rate = learning_rate,
                            beta1 = beta1,
                            beta2 = beta2)
        
    elif name == "nadam":
        beta1 = kwargs.pop("beta1", 0.5)
        beta2 = kwargs.pop("beta2", 0.5)
        optimizer = RMSProp(parameters = model_params, 
                            learning_rate = learning_rate,
                            beta1 = beta1,
                            beta2 = beta2)

    if not optimizer:
        raise ValueError(f"Invalid optimizer name: {name}")
    
    return optimizer
