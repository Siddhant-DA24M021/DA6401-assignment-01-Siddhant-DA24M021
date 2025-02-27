import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_dataset
from dataloader import Dataloader
from linearlayer import LinearLayer
from utils import train_val_split, get_activation_function, get_loss_function, get_initializer
from optimizer import SGD, MomentumGD, NesterovAccGD, RMSProp, Adam, Nadam
from neuralnetwork import NeuralNetwork
import wandb


def sweep_hyperparameters(config = None):

    with wandb.init(config = config):

        config = wandb.config
        wandb.run.name = f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}_op_{config.optimizer}_lr_{config.learning_rate}"


        # C. Log in my details
        wandb.log({"NAME": "SIDDHANT BARANWAL", "ROLL NO.": "DA24M021"})


        # C. Download and load the dataset and discard test data
        (X_train, y_train), (_, _), class_names, image_dict = load_dataset(config.dataset)


        # Log the sample images on wandb report
        wandb.log({"Sample Images": [wandb.Image(img, caption = class_names[class_id]) for class_id, img in image_dict.items()]})


        # D. Data Preprocessing
        # D-1. Flatten the dataset
        X_train = X_train.reshape(X_train.shape[0], -1)

        # D-2. Standardize the dataset between 0-1
        X_train = X_train/255.0

        # D-3. One-Hot Encode the labels for easier processing
        num_classes = len(class_names)
        y_train = np.eye(num_classes)[y_train]


        # E. Create train-val split
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio = 0.1)


        # F. Prepare the Dataloader for batch generation
        train_dataloader = Dataloader(X_train, y_train, batch_size = config.batch_size, shuffle = True)
        val_dataloader = Dataloader(X_val, y_val, batch_size = config.batch_size)


        # G. Create the model architecture and initialize other parameters
        input_size = X_train.shape[1]
        output_size = num_classes
        hidden_size = config.hidden_size

        weight_initializer = get_initializer(config.weight_init)

        layers = []
        hidden_layer1 = LinearLayer(nin = input_size, nout = hidden_size, initializer = weight_initializer)
        activation_layer_1 = get_activation_function(config.activation)
        layers = [hidden_layer1, activation_layer_1]

        for _ in range(config.num_layers - 1):
            layers.append(LinearLayer(nin = hidden_size, nout = hidden_size, initializer = weight_initializer))
            layers.append(get_activation_function(config.activation))

        output_layer = LinearLayer(nin = hidden_size, nout = output_size, initializer = weight_initializer)
        output_activation = get_activation_function("softmax")
        layers.append(output_layer)
        layers.append(output_activation)


        # H. Initialize loss function
        loss_fn = get_loss_function(config.loss)


        # I. Initialize Neural Network model
        model = NeuralNetwork(layers = layers, loss = loss_fn)


        # J. Initialize optimizer
        if config.optimizer == "sgd":
            optimizer = SGD(parameters = model.parameters(), learning_rate = config.learning_rate)
        elif config.optimizer == "momentum":
            optimizer = MomentumGD(parameters = model.parameters(), learning_rate = config.learning_rate, momentum = config.momentum)
        elif config.optimizer == "nag":
            optimizer = NesterovAccGD(parameters = model.parameters(), learning_rate = config.learning_rate, momentum = config.momentum)

        
        # Training Loop
        for epoch in range(config.epochs):
            for data, labels in train_dataloader:
                # Forward pass
                predictions = model.forward(data)
                
                # Compute loss
                loss_value = model.loss(predictions, labels)
                
                # Backward pass
                model.backward()
                
                # Update parameters
                optimizer.step(model.gradients())

        
        # Evaluate on train data
        losses = []
        accuracies = []
        for data, labels in train_dataloader:
            predictions = model.forward(data)
            losses.append(model.loss(predictions, labels))
            accuracies.append(np.sum(np.argmax(predictions, axis = 1) == np.argmax(labels, axis = 1)) / data.shape[0])

        loss = np.mean(losses)
        accuracy = np.mean(accuracies)


        # Evaluate on val data
        losses = []
        accuracies = []
        for data, labels in val_dataloader:
            predictions = model.forward(data)
            losses.append(model.loss(predictions, labels))
            accuracies.append(np.sum(np.argmax(predictions, axis = 1) == np.argmax(labels, axis = 1)) / data.shape[0])

        val_loss = np.mean(losses)
        val_accuracy = np.mean(accuracies)


        # Log the evaluation metrics
        wandb.log({
            "loss": loss,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })


if __name__ == "__main__":

    sweep_config = {"method": "random",
                    "metric": {"name": "val_accuracy", "goal": "maximize"},
                    "parameters": {
                        "epochs": {"values": [5, 10]},
                        "num_layers": {"values": [3, 4, 5]},
                        "hidden_size": {"values": [32, 64, 128]},
                        "weight_decay": {"values": [0, 0.0005, 0.5]},
                        "learning_rate": {"values": [1e-3, 1e-4]},
                        "optimizer": {"values": ["sgd", "momentum"]},
                        "batch_size": {"values": [16, 32, 64]},
                        "weight_init": {"values": ["random"]},
                        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
                        "dataset": {"values": ["fashion_mnist"]},
                        "loss": {"values": ["cross_entropy"]}
                    }}
    
    sweep_id = wandb.sweep(sweep_config, project = "da24m021_da6401_assignment1")
    wandb.agent(sweep_id, function = sweep_hyperparameters, count = 5)