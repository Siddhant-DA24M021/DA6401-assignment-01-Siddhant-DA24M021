import numpy as np

def evaluate(model, data, labels):

    losses = []
    correct = 0
    total = 0

    # Forward pass
    predictions = model.forward(data)
    
    # Compute loss
    loss_value = model.loss(predictions, labels)
    losses.append(loss_value)
    
    # Compute accuracy
    correct += np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1))
    total += data.shape[0]

    return np.mean(losses), correct/total

