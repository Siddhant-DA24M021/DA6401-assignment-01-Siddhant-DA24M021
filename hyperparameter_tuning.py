import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_dataset
from dataloader import Dataloader
from linearlayer import LinearLayer
from utils import train_val_split, get_activation_function, get_loss_function, get_initializer, get_optimizer
from neuralnetwork import NeuralNetwork
from evaluate_model import evaluate
import wandb


def sweep_hyperparameters(config = None):

    with wandb.init(config = config):

        config = wandb.config
        wandb.run.name = f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}_op_{config.optimizer}_lr_{config.learning_rate}"


        # Log in my details
        wandb.config.update({"NAME": "SIDDHANT BARANWAL", "ROLL NO.": "DA24M021"})


        # Download and load the dataset and discard test data
        (X_train, y_train), (_, _), class_names, image_dict = load_dataset(config.dataset)


        # Log the sample images on wandb report
        wandb.log({"Sample Images": [wandb.Image(img, caption = class_names[class_id]) for class_id, img in image_dict.items()]})


        # Data Preprocessing
        # 1. Flatten the dataset
        X_train = X_train.reshape(X_train.shape[0], -1)

        # 2. Standardize the dataset between 0-1
        X_train = X_train/255.0

        # 3. One-Hot Encode the labels for easier processing
        num_classes = len(class_names)
        y_train = np.eye(num_classes)[y_train]


        # Create train-val split
        X_train, y_train, X_val, y_val = train_val_split(X_train, y_train, val_ratio = 0.1)


        # Prepare the Dataloader for batch generation
        train_dataloader = Dataloader(X_train, y_train, batch_size = config.batch_size, shuffle = True)


        # Create the model architecture and initialize other parameters
        input_size = X_train.shape[1]
        output_size = num_classes
        hidden_size = config.hidden_size
        weight_initializer = get_initializer(config.weight_init)
        layers = []

        layers.append(LinearLayer(nin = input_size, nout = hidden_size, initializer = weight_initializer))
        layers.append(get_activation_function(config.activation))

        for _ in range(config.num_layers - 1):
            layers.append(LinearLayer(nin = hidden_size, nout = hidden_size, initializer = weight_initializer))
            layers.append(get_activation_function(config.activation))

        layers.append(LinearLayer(nin = hidden_size, nout = output_size, initializer = weight_initializer))
        layers.append(get_activation_function("softmax"))


        # Initialize loss function
        loss_fn = get_loss_function(config.loss)


        # Initialize Neural Network model
        model = NeuralNetwork(layers = layers, loss = loss_fn)


        # Initialize optimizer (Will only use those arguments which in required for that optimizer)
        optimizer = get_optimizer(config.optimizer,
                                  model.parameters(),
                                  learning_rate = config.learning_rate)

        
        # Training Loop
        for epoch in range(config.epochs):
            losses = []
            correct = 0
            total = 0

            for data, labels in train_dataloader:
                # Forward pass
                predictions = model.forward(data)
                
                # Compute loss
                loss_value = model.loss(predictions, labels)
                

                # Only needed for Nesterov Accelerated Gradient Descent
                if config.optimizer == "nag":
                    optimizer.apply_lookahead()

                # Backward pass
                model.backward()
                
                # Update parameters
                optimizer.step(model.gradients())

                losses.append(loss_value)
                correct += np.sum(np.argmax(predictions, axis = 1) == np.argmax(labels, axis = 1))
                total += data.shape[0]

        
            # Evaluate on train data
            loss, accuracy = np.mean(losses), correct/total
            print(f"Epoch {epoch + 1: 4}:     Training Loss - {loss : 10.5f}   &   Training Accuracy - {accuracy: 10.5f}")

            # Evaluate on val data
            val_loss, val_accuracy = evaluate(model, X_val, y_val)

            # Log the evaluation metrics
            wandb.log({
                "epoch": epoch + 1,
                "loss": loss,
                "accuracy": accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            })


if __name__ == "__main__":

    sweep_config = {"method": "random",
                    "metric": {"name": "val_accuracy", "goal": "maximize"},
                    "parameters": {
                        "epochs": {"values": [10]},
                        "num_layers": {"values": [3, 4, 5]},
                        "hidden_size": {"values": [32, 64, 128]},
                        "weight_decay": {"values": [0, 0.0005, 0.5]},
                        "learning_rate": {"values": [1e-1, 1e-2, 1e-3, 1e-4]},
                        "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop"]},
                        "batch_size": {"values": [16, 32, 64]},
                        "weight_init": {"values": ["random", "Xavier"]},
                        "activation": {"values": ["sigmoid", "tanh", "ReLU"]},
                        "dataset": {"values": ["fashion_mnist"]},
                        "loss": {"values": ["cross_entropy"]}
                    }}
    
    sweep_id = wandb.sweep(sweep_config, project = "da24m021_da6401_assignment1")
    wandb.agent(sweep_id, function = sweep_hyperparameters, count = 10)