import numpy as np
import sys
from neural_network.nn_functions import *

LAYERS = []

class Layer:
    def __init__(self, dim : int):
        self.dim = dim

        # Instantiate neurons and biases matrices
        self.neurons = np.zeros((dim, 1))
        self.biases = np.zeros((dim, 1))

        # Create m x n (L2 neuron count x L1 neuron count) weights matrix using He init.
        if LAYERS:
            self.previous_layer = LAYERS[-1]
            previous_dim = self.previous_layer.dim
            self.weights = He_initialization(dim, previous_dim)

        # If this is L1 then there will be no weights matrix
        else:
            self.previous_layer, self.weights = None, None

        # Append to layer sequence (to be able to reference previous layers)
        LAYERS.append(self)

    def compute_neurons(self):
        # L1 should be from dataset and never computed
        if not self.previous_layer:
            raise Exception("Cannot compute L1 neurons.")

        # Multiply weights and input neurons
        weighted_sum = self.get_weighted_sum()
        # Apply ReLU for all layers except last, which uses Softmax
        if self == LAYERS[-1]:
            activated_sum = softmax(weighted_sum)
        else:
            activated_sum = ReLU(weighted_sum)
        # Update this layers' neurons
        self.update_neurons(activated_sum)

    def get_weighted_sum(self):
        return (self.weights @ self.previous_layer.neurons) + self.biases

    def update_neurons(self, neurons : list | np.ndarray):
        # Make n x 1 column vector from list. Reject if the dimension is incorrect
        if not isinstance(neurons, np.ndarray):
            neurons = np.array(neurons).reshape(-1, 1)
        if neurons.shape[0] != self.dim:
            neurons = neurons.reshape(-1, 1)
        if neurons.shape != (self.dim, 1):
            raise Exception("Incorrect dimension.")

        self.neurons = neurons

    def update_params(self, dW : np.ndarray, db : np.ndarray, step : float):
        self.weights = self.weights - (dW * step)
        self.biases = self.biases - (db * step)



def He_initialization(current_dim, previous_dim):
    seed = np.random.randint(0, 2147483647)
    np.random.seed(seed)
    std_dev = np.sqrt(2.0 / previous_dim)
    weights = np.random.normal(loc = 0.0, scale = std_dev, size = (current_dim, previous_dim))
    return weights


