




# DONT FORGET TO MAKE GITHUB REPO PUBLIC BEFORE SUBMITTING










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

# Hyperparameter Tuning
python hyperparameter_tuning.py

# Using the model
python train.py --dataset fashion_mnist --epochs 10 --batch_size 128 --hidden_size 64 --num_layers 3 --activation tanh --optimizer adam --learning_rate 0.001 --loss cross_entropy --weight_init Xavier