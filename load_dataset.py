import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, fashion_mnist



# Load the dataset and plot images from each class
def load_dataset(dataset_name = "fashion_mnist", wandb_log = True):

    if dataset_name == "mnist": 
        # Download mnist dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Class names in MNIST
        class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    else:
        # Download fashion_mnist dataset
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        # Class names in Fashion-MNIST
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Select sample images
    classes = np.unique(y_train)
    sample_images = [X_train[np.where(y_train == cls)[0][0]] for cls in classes]

    # Plot sample images
    plt.figure(figsize = (10, 5))
    for i in range(len(classes)):
        plt.subplot(2, 5, i+1)
        plt.imshow(sample_images[i])
        plt.title(class_names[i])
        plt.axis(False)

    if wandb_log:
        # Initialize and log to wandb
        wandb.init(project="da24m021_da6401_assignment1")
        wandb.log({"Question_1_fashion_mnist_samples": wandb.Image(plt)})
        wandb.finish()
    else:
        plt.show() # only show if not logging to wandb.

    return (X_train, y_train), (X_test, y_test), class_names

if __name__ == "__main__":
    load_dataset()