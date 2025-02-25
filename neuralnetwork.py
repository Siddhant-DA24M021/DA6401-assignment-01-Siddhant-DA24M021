class NeuralNetwork:
    def __init__(self, layers, loss):
        self.layers = layers
        self.loss = loss

    def forward(self, x): # Forward Propagation
        self.output = x
        for layer in self.layers:
            self.output = layer(self.output)
        return self.output


    def backward(self): # BackPropagation Step
        grad_output = self.loss.backward() 

        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output) # This will populate the gradients of all the model parameters

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def gradients(self):
        grads = []
        for layer in self.layers:
            grads.extend(layer.gradients()) # Return all the model parameter's gradients
        return grads
    
    def prediction(self, x):
        self.output = x
        for layer in self.layers:
            self.output = layer(self.output)
        return self.output