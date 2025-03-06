import numpy as np
from load_dataset import load_dataset
from dataloader import Dataloader
from linearlayer import LinearLayer
from utils import get_activation_function, get_loss_function, get_initializer, get_optimizer
from neuralnetwork import NeuralNetwork
from evaluate_model import evaluate
from arguments_parser import parse_arguments
from log_predicted_images import log_predicted_images
import wandb


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
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], -1)

    # F-2. Standardize the dataset between 0-1
    X_train_std = X_train_reshaped/255.0
    X_test_std = X_test_reshaped/255.0

    # F-3. One-Hot Encode the labels for easier processing
    num_classes = len(class_names)
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh = np.eye(num_classes)[y_test]




    # G. Prepare the Dataloader for batch generation
    train_dataloader = Dataloader(X_train_std, y_train_oh, batch_size = args.batch_size, shuffle = True)


    # H. Create the model architecture and initialize other parameters
    input_size = X_train_std.shape[1]
    output_size = num_classes
    hidden_size = args.hidden_size
    weight_initializer = get_initializer(args.weight_init)
    layers = []

    # Hidden Layer 1
    layers.append(LinearLayer(nin = input_size, nout = hidden_size, initializer = weight_initializer))
    layers.append(get_activation_function(args.activation))

    # Hidden Layers
    for _ in range(args.num_layers - 1):
        layers.append(LinearLayer(nin = hidden_size, nout = hidden_size, initializer = weight_initializer))
        layers.append(get_activation_function(args.activation))

    # Output Layer 
    layers.append(LinearLayer(nin = hidden_size, nout = output_size, initializer = weight_initializer))
    layers.append(get_activation_function(args.output_activation))


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

            # Only needed for Nesterov Accelerated Gradient Descent
            if args.optimizer == "nag":
                optimizer.apply_lookahead()
            
            # Backward pass (Backpropagation)
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
        test_loss, test_accuracy = evaluate(model, X_test_std, y_test_oh)


        # N. Log details on wandb
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })

    print("--------------------------------------------------------------------------------------------------")


    # M. Log sample predictions in wandb
    log_predicted_images(model, X_test, X_test_std, y_test, class_names, 10)


    # N. Create and log a confusion matrix
    prediction_probs = model.forward(X_test_std)
    predictions = np.argmax(prediction_probs, axis = 1)
    actual_label = y_test

    wandb.log({"Test Data Confusion Matrix": wandb.plot.confusion_matrix(preds = predictions, y_true = actual_label, class_names = class_names)})



    # O. Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()