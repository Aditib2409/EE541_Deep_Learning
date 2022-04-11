import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from collections import defaultdict
from tqdm import tqdm
import pickle as pkl


def extract_files(path):
    """
    :param path: path to the file to be extracted
    :return: extracted file
    """
    data = h5py.File(path, 'r')
    filename = os.path.splitext(os.path.basename(path))[0]
    string = filename.split("_")[1]

    x = data['xdata']
    y = data['ydata']

    return x, y


def split_train_validation(x, y, split):
    if split:
        data = np.concatenate((x, y), axis=1)
        train = data[0:50000, :]
        validation = data[50000:60000, :]


        train_x = train[:, 0:784]
        train_y = train[:, 784:794]

        val_x = validation[:, 0:784]
        val_y = validation[:, 784:794]
        
        return train_x, train_y, val_x, val_y


def shuffle_dataset(x, y, shuffle):
    if shuffle:
        data = np.concatenate((x, y), axis=1)
        np.random.shuffle(data) # shuffling the train dataset
        train_x = data[:, 0:784]
        train_y = data[:, 784:794]

        return train_x, train_y


# NN architecture
nn_architecture = [
    {"input_dim": 784, "output_dim": 200, "activation": "RELU"},
    {"input_dim": 200, "output_dim": 100, "activation": "RELU"},
    {"input_dim": 100, "output_dim": 10, "activation": "SOFTMAX"},
]


def initializing_layers(nn_archi):
    np.random.seed(42)
    parameters = defaultdict(list)  # creates an empty list when KeyError

    # iterating over each layer of the neural network
    for index, layer in enumerate(nn_archi):
        layer_index = index+1  # Since layer number starts from 1

        layer_input_dim = layer["input_dim"]
        layer_output_dim = layer["output_dim"]

        #  Initializing the parameters for each layer
        parameters['W' + str(layer_index)] = np.random.normal(loc=0, scale=1/layer_output_dim, size=(layer_output_dim, layer_input_dim))
        parameters['b' + str(layer_index)] = np.random.normal(size=(layer_output_dim,))

    return parameters


def initial_gradients_layers(nn_archi):
    np.random.seed(42)
    gradients_intial = defaultdict(list) # creates an empty list when KeyError

    # iterating over each layer of the neural network
    for index, layer in enumerate(nn_archi):
        layer_index = index+1  # Since layer number starts from 1

        layer_input_dim = layer["input_dim"]
        layer_output_dim = layer["output_dim"]

        #  Initializing the parameters for each layer
        gradients_intial['dw' + str(layer_index)] = np.zeros((layer_output_dim, layer_input_dim))
        gradients_intial['db' + str(layer_index)] = np.zeros((layer_output_dim,))

    return gradients_intial


def ReLU(x):
    func = np.maximum(0, x)
    return func


def tanh(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    func = numerator/denominator
    func_deri = 1 - func**2
    return func, func_deri


def softmax(x):
    func = np.exp(x)
    func_sum = np.sum(func)
    return func/func_sum


def ReLU_derivative(y):
    dfunc = np.array(y)
    return np.where(dfunc <= 0, 0, 1)


def forward_prop(x, parameters, nn_archi):
    layer_outputs = defaultdict(list)
    A = x

    for index, layer in enumerate(nn_archi):
        layer_index = index+1
        w = parameters['W' + str(layer_index)]
        b = parameters['b' + str(layer_index)]
        A_prev = A # input to layer
        activation_function_str = layer["activation"]

        if activation_function_str == "RELU":
            activation_function = ReLU
        elif activation_function_str == "SOFTMAX":
            activation_function = softmax

        # Computing the forward propagation for a single layer
        y = np.matmul(w, A_prev) + b  # This is linear output
        A = activation_function(y)  # This is output to layer (with activation)

        layer_outputs['A_prev' + str(index)] = A_prev  # non-linear output of [l-1]
        layer_outputs['y' + str(layer_index)] = y  # linear output of [l]

    return A, layer_outputs


# Computing cost function and its derivative (delta)
def cross_entropy(y_pred, y_actual):  # each image
    cost = -np.dot(y_actual, np.log(y_pred))  # scalar
    cost_derivative = y_pred - y_actual # (10,)
    return cost, cost_derivative


def back_prop(derivative_cost, nn_archi, layer_outputs, parameters, gradients_total):
    delta_layer = derivative_cost  # delta 3

    gradients = defaultdict(list)

    for index, layer in reversed(list(enumerate(nn_archi))):
        layer_index = index + 1
        A_prev = layer_outputs['A_prev' + str(index)]
        y = layer_outputs['y' + str(layer_index)]

        w = parameters['W' + str(layer_index)]
        b = parameters['b' + str(layer_index)]

        # activation_function_str = layer["activation"]
        dA_prev = ReLU_derivative(A_prev)
        delta_layer_prev = dA_prev * np.matmul(w.T, delta_layer)

        dw = np.matmul(delta_layer.reshape(-1, 1), A_prev.reshape(1, -1))
        db = delta_layer

        delta_layer = delta_layer_prev

        gradients['dw' + str(layer_index)] = dw
        gradients['db' + str(layer_index)] = db

        gradients_total['dw' + str(layer_index)] += dw
        gradients_total['db' + str(layer_index)] += db

    return gradients, gradients_total


def optimizer(parameters, gradients, lr, nn_archi): # updating using SGD

    for index, layer in enumerate(nn_archi):
        layer_index = index + 1

        parameters['W' + str(layer_index)] = parameters['W' + str(layer_index)] - lr*gradients['dw' + str(layer_index)]
        parameters['b' + str(layer_index)] = parameters['b' + str(layer_index)] - lr*gradients['db' + str(layer_index)]

    return parameters


def predict(x, nn_archi, parameters):
    layer_outputs = defaultdict(list)
    A = x

    for index, layer in enumerate(nn_archi):
        layer_index = index + 1
        w = parameters['W' + str(layer_index)]
        b = parameters['b' + str(layer_index)]
        A_prev = A
        activation_function_str = layer["activation"]

        if activation_function_str == "RELU":
            activation_function = ReLU
        elif activation_function_str == "SOFTMAX":
            activation_function = softmax

        # Computing the forward propagation for a single layer
        y = np.matmul(w, A_prev) + b
        A = activation_function(y)

        layer_outputs['A_prev' + str(index)] = A_prev  # non-linear output of [l-1]
        layer_outputs['y' + str(layer_index)] = y

    return A


def accuracy(x, y_actual, nn_archi, parameters):
    correct = 0
    for i in range(x.shape[0]):
        y_pred = predict(x[i, :], nn_archi, parameters)
        if np.argmax(y_pred) == np.argmax(y_actual[i, :]):
            correct += 1
    correct_percent = (correct/x.shape[0])*100
    return correct, correct_percent


def fit(X, Y, epochs, nn_archi, lr, batch_size):

    lr_decay1 = lr/2
    lr_decay2 = lr/4
    parameters = initializing_layers(nn_archi)
    num_batches = int(X.shape[0]/batch_size)
    accuracy_array = np.zeros([epochs, ])
    for epoch in range(epochs):
        X, Y = shuffle_dataset(X, Y, shuffle=True)
        if epoch > 20:
            lr = lr_decay1
        elif epoch > 40:
            lr = lr_decay2
        else:
            lr = lr
        for i in tqdm(range(num_batches)):  # Each batch
            batch_cost = 0
            gradients_total = initial_gradients_layers(nn_archi)
            for j in range(batch_size):  # Each data in a batch

                k = i*batch_size + j

                A_pred, outputs_layers = forward_prop(X[k, :], parameters, nn_archi) # single image

                cost, cost_derivative = cross_entropy(A_pred, Y[k, :]) # last layer delta[last_layer]
                batch_cost += cost

                gradients, average_gradients = back_prop(cost_derivative, nn_archi, outputs_layers, parameters, gradients_total)

            for key in average_gradients:
                average_gradients[key] = np.array(average_gradients[key])/batch_size

            parameters = optimizer(parameters, average_gradients, lr, nn_archi)

        correct, percent = accuracy(X, Y, nn_architecture, parameters)
        print(f'Epoch:{epoch+1} | No. of correct predictions = {correct} | accuracy = {percent:.3f}%')
        print(f'Total cost after each batch: {batch_cost}')
        accuracy_array[epoch] = percent
    return parameters, accuracy_array


if __name__ == "__main__":
    learning_rate = 0.1
    num_epochs = 50
    X, Y = extract_files("mnist_traindata.hdf5")
    # split_train_validation(X, Y, split=True)
    x_train, y_train, x_val, y_val = split_train_validation(X, Y, split=True)

    optimal_parameters, accuracy_rate = fit(x_train, y_train, num_epochs, nn_architecture, learning_rate, 500)

    with open('weights_01_relu.pkl', 'wb') as file:
        pkl.dump(optimal_parameters, file)

    epoch_array = np.arange(0, num_epochs, 1)
    print(accuracy_rate)
    plt.figure()
    plt.plot(epoch_array, accuracy_rate)
    plt.title(f'Training accuracy')
    plt.xlabel(f'epochs')
    plt.ylabel(f'percentage of correct predictions(%)')
    plt.savefig(f'figures/colab1_01_relu.png')
    plt.show()

