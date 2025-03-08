



# GITHUB LINK:- https://github.com/Siddhant-DA24M021/DA6401-assignment-01-Siddhant-DA24M021.git
# REPORT LINK:- https://api.wandb.ai/links/da24m021-indian-institute-of-technology-madras/tchaabjk





# DA6401-assignment-01-Siddhant-DA24M021
DA24M021 Siddhant Baranwal Assignment 1 of the course DA6401: Introduction to Deep Learning


# Dependencies
1. numpy
2. argparse
3. keras    # For data import only
4. wandb

# Install Dependencies
pip install numpy wandb keras

# Overview
This project implements a simple neural network from scratch using Numpy and python. It includes custom implementations of layers, activation functions, loss functions, optimizers, and hyperparameter tuning.


# Features
1. Fully connected neural network implementation
2. Custom activation functions: ReLU, Sigmoid, Tanh, Softmax, Identity
3. Custom loss functions: Cross-Entropy, Mean Squared Error
4. Optimizers: SGD, Momentum, NAG, RMSProp, Adam, Nadam
5. Weight initializers: Random Normal, Xavier, Uniform
6. Dataset loading and preprocessing (supports MNIST and Fashion-MNIST)
7. Hyperparameter tuning using Weights & Biases (wandb)

# File Structure
1. activation.py: Implements activation functions
2. arguments_parser.py: Parses arguments passed through command line 
3. dataloader.py: Prepares batches of data for the neural network
4. evaluate_model.py: Calculates the accuracy of the model 
5. hyperparameter_tuning.py: Performs hyperparameter tuning using Weights & Biases sweep
6. initializers.py: Implements different weight initialization methods
7. linearlayer.py: Implements the linear layer for the network
8. load_dataset.py: Loads and preprocesses dataset (MNIST, Fashion-MNIST)
9. log_predicted_images.py: logs samples of predicted images to wandb
10. loss.py: Implements loss functions (Cross-Entropy, MSE)
11. neuralnetwork.py: Defines the neural network architecture
12. optimizer.py: Implements optimization algorithms
13. train.py: Trains the neural network and logs metrics using wandb
14. utils.py: Helper functions for activation, loss, optimizer selection, and dataset splitting
15. cross_entropy_vs_mean_squared_loss.py: File to compare cross-entropy loss vs mean-squared-error loss

# Hyperparameter Tuning
To run hyperparameter tuning using Weights & Biases sweeps, execute the following command:
python hyperparameter_tuning.py

# Using the model
python train.py --dataset fashion_mnist --epochs 15 --batch_size 128 --hidden_size 64 --num_layers 3 --activation tanh --optimizer adam --learning_rate 0.001 --loss cross_entropy --weight_init Xavier


# Available command line arguments
1.  -wp, --wandb_project         Project name used to track experiments in Weights & Biases dashboard (default: da24m021_da6401_assignment1)
2.  -we, --wandb_entity          Wandb Entity used to track experiments in the Weights & Biases dashboard. (default: da24m021-indian-institute-of-technology-madras)
3.  -d, --dataset                Dataset to use (choices: mnist, fashion_mnist, default: fashion_mnist)
4.  -e, --epochs                 Number of epochs to train neural network. (default: 15)
5.  -b, --batch_size             Batch size used to train neural network. (default: 128)
6.  -l, --loss                   Loss function (choices: mean_squared_error, cross_entropy, default: cross_entropy)
7.  -o, --optimizer              Optimizer (choices: sgd, momentum, nag, rmsprop, adam, nadam, default: adam)
8.  -lr, --learning_rate         Learning rate used to optimize model parameters (default: 0.001)
9.  -m, --momentum               Momentum used by momentum and nag optimizers. (default: 0.5)
10.  -beta, --beta                Beta used by rmsprop optimizer (default: 0.5)
11.  -beta1, --beta1              Beta1 used by adam and nadam optimizers. (default: 0.9)
12.  -beta2, --beta2              Beta2 used by adam and nadam optimizers. (default: 0.999)
13.  -eps, --epsilon              Epsilon used by optimizers. (default: 1e-08)
14.  -w_d, --weight_decay         Weight decay used by optimizers. (default: 0.0)
15.  -w_i, --weight_init          Weight initialization method (choices: random, Xavier, default: Xavier)
16.  -nhl, --num_layers           Number of hidden layers used in feedforward neural network. (default: 3)
17.  -sz, --hidden_size          Number of hidden neurons in a feedforward layer. (default: 64)
18.  -a, --activation            Activation function (choices: identity, sigmoid, tanh, ReLU, default: tanh)
19.  -oa, --output_activation     Output activation function (choices: softmax, identity, sigmoid, tanh, ReLU, default: softmax)


# Self Declaration
I, Siddhant Baranwal DA24M021, swear on my honour that I have written the code and the report by myself and have not copied it from the internet or other students.