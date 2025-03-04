import argparse

# Parse command line arguments
def parse_arguments():


    parser = argparse.ArgumentParser()

    parser.add_argument("-wp", "--wandb_project", type = str, default = "da24m021_da6401_assignment1", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type = str, default = "da24m021-indian-institute-of-technology-madras", help="Wandb Entity used to track experiments in the Weights & Biases dashboard.")
    parser.add_argument("-d", "--dataset", type = str, choices = ["mnist", "fashion_mnist"], default = "fashion_mnist", help = "Dataset to use")
    parser.add_argument("-e", "--epochs", type = int, default = 10, help = "Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", type = int, default = 32, help = "Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", type = str, choices = ["mean_squared_error", "cross_entropy"], default = "cross_entropy", help = "Loss function")
    parser.add_argument("-o", "--optimizer", type = str, choices = ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default = "sgd", help = "Optimizer")
    parser.add_argument("-lr", "--learning_rate", type = float, default = 0.01, help = "Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type = float, default = 0.5, help = "Momentum used by momentum and nag optimizers.")
    parser.add_argument("-beta", "--beta", type = float, default = 0.5, help = "Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type = float, default = 0.9, help = "Beta1 used by adam and nadam optimizers.")
    parser.add_argument("-beta2", "--beta2", type = float, default = 0.999, help = "Beta2 used by adam and nadam optimizers.")
    parser.add_argument("-eps", "--epsilon", type = float, default = 1e-8, help = "Epsilon used by optimizers.")
    parser.add_argument("-w_d", "--weight_decay", type = float, default = 0.0, help = "Weight decay used by optimizers.")
    parser.add_argument("-w_i", "--weight_init", type = str, choices = ["random", "Xavier"], default = "random", help = "Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type = int, default = 4, help = "Number of hidden layers used in feedforward neural network.")
    parser.add_argument("-sz", "--hidden_size", type = int, default = 64, help = "Number of hidden neurons in a feedforward layer.")
    parser.add_argument("-a", "--activation", type = str, choices = ["identity", "sigmoid", "tanh", "ReLU"], default = "sigmoid", help = "Activation function")
    parser.add_argument("-oa", "--output_activation", type = str, choices = ["softmax", "identity", "sigmoid", "tanh", "ReLU"], default = "softmax", help = "Output activation function")

    return parser.parse_args()