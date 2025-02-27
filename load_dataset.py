import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist, fashion_mnist



# Load the dataset and plot images from each class
def load_dataset(dataset_name = "fashion_mnist"):

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
    image_dict = {class_name: X_train[np.where(y_train == class_name)[0][0]] for class_name in classes}

    return (X_train, y_train), (X_test, y_test), class_names, image_dict

if __name__ == "__main__":
    load_dataset()