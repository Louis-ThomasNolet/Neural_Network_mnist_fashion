import numpy as np
import nnfs
import os
import cv2
import matplotlib.pyplot as plt
from Network import Model, Layers, Optimizers, Loss, Activations, Accuracy

nnfs.init()

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create lists for samples and labels
    X = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(
                        os.path.join(path, dataset, label, file),
                        cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            X.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(X), np.array(y).astype('uint8')


# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return X, y, X_test, y_test

# Create the dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

#Shuffle the training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# Scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

# Instantiate the model
model = Model.Model()

# Add layers
model.add(Layers.Layer_Dense(X.shape[1], 128))
model.add(Activations.Activation_ReLU())
model.add(Layers.Layer_Dense(128, 128))
model.add(Activations.Activation_ReLU())
model.add(Layers.Layer_Dense(128, 10))
model.add(Activations.Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss.Loss_CategoricalCrossentropy(),
    optimizer=Optimizers.Optimizer_Adam(decay=1e-7),
    accuracy=Accuracy.Accuracy_Categorical()
)


# Finalize the model
model.finalize()

model.save('fashion_mnist.model')