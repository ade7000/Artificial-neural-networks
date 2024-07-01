import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def activation(neuron):
    # Implementing sigmoid function
    return 1 / (1 + np.exp(-neuron))

def calculate_neuron(input, weights, bias):
    return np.dot(weights, input) + bias

def mean_squared_error(y_predicted, y_true):
    return 2 * (y_predicted - y_true) / np.size(y_true)

def adjust_weights(input, error, learning_rate):
     return weights - learning_rate * np.dot(error, input.T)

# Reading the dataset and labels
testData = pd.read_csv('mnist_numbers_8x8_0_and_1.csv')
# In this case, the images are stored in a CSV file, each image in a row, with the label in the first column and then each pixel in a separate column
X = testData.iloc[:, 1:].values / 255
Y = testData.iloc[:, 0].values
X = X.reshape(X.shape[0], 8 * 8, 1)
Y = np_utils.to_categorical(Y) # This function transforms a vector of the form [2, 3, 1] into a matrix of the form [[0,0,1,0],[0,0,0,1],[0,1,0,0]]
# This is a common practice in building neural networks for calculating the error of each label.
Y = Y.reshape(Y.shape[0], 2, 1)
# Splitting the dataset into two parts - one for training and one for testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

input_size = len(x_train[0])
output_size = 2 # Only two labels, one for 0 and one for digit 1
weights = np.random.randn(output_size, input_size) # Initializing the weights with random values
bias = np.random.randn(output_size, 1)

learning_rate = 0.1
epochs = 100 # How many times to analyze the data for the network to learn
for e in tqdm(range(epochs)):
    for input, label in zip(x_train, y_train):
        # Calculating neuron value
        neuron = calculate_neuron(input, weights, bias)
        activated_neuron = activation(neuron)
        # Calculating error
        error = np.array(mean_squared_error(activated_neuron, label))
        # Adjusting weights
        weights = adjust_weights(input, error, learning_rate)

# Testing the network
correct = 0
for input, label in tqdm(zip(x_test, y_test)):
    # Calculating neuron value
    neuron = calculate_neuron(input, weights, bias)
    activated_neuron = activation(neuron)
    # Checking if the network identified the label correctly
    if np.argmax(activated_neuron) == np.argmax(label):
        correct += 1

# Calculating the network accuracy as the number of correctly identified labels divided by the total number of images
print("Accuracy is ", str(correct / len(x_test) * 100) + "%")
