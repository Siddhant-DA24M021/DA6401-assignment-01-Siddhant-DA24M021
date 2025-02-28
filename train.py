import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_dataset
from dataloader import Dataloader
from linearlayer import LinearLayer
from utils import get_activation_function, get_loss_function, get_initializer, get_optimizer
from neuralnetwork import NeuralNetwork
from evaluate_model import evaluate
import argparse
import wandb


# Parse command line arguments
def parse_arguments():


    parser = argparse.ArgumentParser()

    parser.add_argument("-wp", "--wandb_project", type = str, default = "da24m021_da6401_assignment1", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type = str, default = "da24m021-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type = str, choices = ["mnist", "fashion_mnist"], default = "fashion_mnist", help = "Dataset to use")
    parser.add_argument("-e", "--epochs", type = int, default = 10, help = "Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type = int, default = 4, help = "Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", type = str, choices = ["mean_squared_error", "cross_entropy"], default = "cross_entropy", help = "Loss function")
    parser.add_argument("-o", "--optimizer", type = str, choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default = "sgd", help = "Optimizer")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 0.1, help = "Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type = float, default = 0.5, help = "Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type = float, default = 0.5, help = "Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type = float, default = 0.5, help = "Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type = float, default = 0.5, help = "Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type = float, default = 1e-6, help = "Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type = float, default = 0.0, help = "Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type = str, choices = ["random", "Xavier"], default = "random", help = "Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type = int, default = 4, help = "Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type = int, default = 64, help = "Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", type = str, choices = ["identity", "sigmoid", "tanh", "ReLU"], default = "sigmoid", help = "Activation function")
    parser.add_argument("-oa", "--output_activation", type = str, choices = ["softmax", "identity", "sigmoid", "tanh", "ReLU"], default = "softmax", help = "Output activation function")

    return parser.parse_args()

# Main Function
def main():

    # A. Parse Arguments
    args = parse_arguments()


    # B. Initialize wandb project
    wandb.init(project = args.wandb_project, entity = args.wandb_entity) 


    # C. Log in my details
    wandb.config.update({"NAME": "SIDDHANT BARANWAL", "ROLL NO.": "DA24M021"})


    # D. Download and load the dataset
    (X_train, y_train), (X_test, y_test), class_names, image_dict = load_dataset(args.dataset)


    # E. Log the sample images on wandb report
    wandb.log({"Sample Images": [wandb.Image(img, caption = class_names[class_id]) for class_id, img in image_dict.items()]})


    # F. Data Preprocessing
    # F-1. Flatten the dataset
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # F-2. Standardize the dataset between 0-1
    X_train = X_train/255.0
    X_test = X_test/255.0

    # F-3. One-Hot Encode the labels for easier processing
    num_classes = len(class_names)
    y_train = np.eye(num_classes)[y_train]
    y_test = np.eye(num_classes)[y_test]




    # G. Prepare the Dataloader for batch generation
    train_dataloader = Dataloader(X_train, y_train, batch_size = args.batch_size, shuffle = True)
    test_dataloader = Dataloader(X_test, y_test)


    # H. Create the model architecture and initialize other parameters
    input_size = X_train.shape[1]
    output_size = num_classes
    hidden_size = args.hidden_size

    weight_initializer = get_initializer(args.weight_init)

    layers = []
    hidden_layer1 = LinearLayer(nin = input_size, nout = hidden_size, initializer = weight_initializer)
    activation_layer_1 = get_activation_function(args.activation)
    layers = [hidden_layer1, activation_layer_1]

    for _ in range(args.num_layers - 1):
        layers.append(LinearLayer(nin = hidden_size, nout = hidden_size, initializer = weight_initializer))
        layers.append(get_activation_function(args.activation))

    output_layer = LinearLayer(nin = hidden_size, nout = output_size, initializer = weight_initializer)
    output_activation = get_activation_function(args.output_activation)
    layers.append(output_layer)
    layers.append(output_activation)


    # I. Initialize loss function
    loss_fn = get_loss_function(args.loss)


    # J. Initialize Neural Network model
    model = NeuralNetwork(layers = layers, loss = loss_fn)


    # K. Initialize optimizer (Will only use those arguments which in required for that optimizer)
    optimizer = get_optimizer(args.optimizer,
                              model.parameters(),
                              learning_rate = args.learning_rate,
                              momentum = args.momentum,
                              beta = args.beta,
                              beta1 = args.beta1,
                              beta2 = args.beta2,
                              epsilon = args.epsilon,
                              weight_decay = args.weight_decay)

    
    # L. Training Loop
    print("--------------------------------------------------------------------------------------------------")
    for epoch in range(args.epochs):
        losses = []
        correct = 0
        total = 0

        for data, labels in train_dataloader:
            # Forward pass
            predictions = model.forward(data)
            
            # Compute loss
            loss_value = model.loss(predictions, labels)
            
            # Backward pass
            model.backward()
            
            # Update parameters
            optimizer.step(model.gradients())
            losses.append(loss_value)

            correct += np.sum(np.argmax(predictions, axis = 1) == np.argmax(labels, axis = 1))
            total += data.shape[0]

        train_loss = np.mean(losses)
        train_accuracy = correct/total
        print(f"Epoch {epoch + 1: 4}:      Training Loss - {train_loss : 10.5f}   &   Training Accuracy - {train_accuracy: 10.5f}")
        
        # M. Evaluation metric on test data
        test_loss, test_accuracy = evaluate(model, test_dataloader)


        # N. Log details on wandb
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

    print("--------------------------------------------------------------------------------------------------")


    # O. Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()