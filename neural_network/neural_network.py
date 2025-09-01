import numpy as np

import hyperparameters_flags as hp
from neural_network.nn_functions import *

"""
neural_network.py represents the neural network.
    -Holds layers, neurons, and parameters
"""

# LAYERS - project-wide container for all layer objects
# - Should probably be a class object but trying to refactor broke everything
LAYERS = []

class Layer:
    """
    Layer represents a single layer, holding attached parameters and references
    to the previous layer. 
        - Next layer references not included because only use is for backpropagation
        - First layer holds no parameters because its neurons are not computed
        - All neurons are computed by applying its layer's attached parameters  
        to previous layer's neurons """""

    # Initialize each layer - dim represents the neuron count
    def __init__(self, dim : int):
        # self.dim represents the layer's dimension (neuron count)
        self.dim = dim

        # Instantiate neurons and bias vectors based on neuron count at 0
        self.neurons = np.zeros((dim, 1))
        self.biases = np.zeros((dim, 1))

        # If this is the first layer, there are no weights parameters or previous layer
        if not LAYERS:
            self.previous_layer, self.weights = None, None

        else:
            # Assign previous layer to current last Layer in LAYERS container
            self.previous_layer = LAYERS[-1]
            # Get last layer's neuron count to instantiate right-sized weights matrix
            previous_dim = self.previous_layer.dim

            """
            Note on fixed seeds:
                1. The first layer generates a seed based on the given seed, which is used for
                He initialization.
                2. All succeeding layers then generate new seeds based on the previous seed and 
                initialize based on their new seed.
                
                This is deterministic, so a fixed initial seed will always reproduce 
                an identical process, ensuring testing is possible. """
            # If hp.LOAD_SEED flag = True and the current layer is the first,
            # use the given seed from hp.SEED
            if hp.LOAD_SEED and len(LAYERS) == 1:
                # Initialize the weights using random He initialization
                self.weights, seed = He_initialization(dim, previous_dim, hp.SEED)

            # If this layer isn't the first or hp.LOAD_SEED is False, use
            # a seed given by random.randint to initialize the weights.
            else:
                self.weights, seed = He_initialization(dim, previous_dim)

                # Export initially generated seed to hp.SEED for persistent record
                if len(LAYERS) == 1:
                    hp.SEED = seed

        # Add initialized layer to LAYERS
        LAYERS.append(self)

    def compute_neurons(self):
        # Compute the neurons of the "self" layer.

        # L1 should be from dataset and never computed
        if not self.previous_layer:
            raise Exception("Cannot compute L1 neurons.")

        # Get weighted sum from previous layer's neurons
        weighted_sum = self.get_weighted_sum()

        # All layers but the last use ReLU activation. Last layer uses Softmax.
        if self == LAYERS[-1]:
            activated_sum = softmax(weighted_sum)
        else:
            activated_sum = ReLU(weighted_sum)

        # Finally, update the 'self' layer's neurons.
        self.update_neurons(activated_sum)

    def get_weighted_sum(self) -> np.ndarray:
        # Get raw weighted sum based on parameters attached to current layer
        # and previous layer's neurons

        # Formula is weights matrix * previous layer's neurons + bias vector
        return (self.weights @ self.previous_layer.neurons) + self.biases

    def update_neurons(self, neurons : list | np.ndarray):
        # Ensure that neurons being passed are of correct dimension and format pre-update

        # Make n x 1 column vector from list. Reject if the dimension is incorrect
        if not isinstance(neurons, np.ndarray):
            neurons = np.array(neurons).reshape(-1, 1)
        if neurons.shape[0] != self.dim:
            neurons = neurons.reshape(-1, 1)
        if neurons.shape != (self.dim, 1):
            raise Exception("Incorrect dimension.")

        # Update the neurons
        self.neurons = neurons

    def update_params(self, dW : np.ndarray, db : np.ndarray):
        # Update 'self' layer's weights & biases based on passed gradients
        # & learning rate specified in hp.LEARNING_RATE
        self.weights = self.weights - (dW * hp.LEARNING_RATE)
        self.biases = self.biases - (db * hp.LEARNING_RATE)



def He_initialization(current_dim : int, previous_dim : int, seed = 0) -> tuple[np.ndarray, int]:
    """
    He initialization to initialize weights consistently w/ appropriate distribution """""

    # If no seed is passed, generate seed randomly
    if not seed:
        seed = np.random.randint(1, 2147483647)

    # Generate random seed
    np.random.seed(seed)

    # He initialization is a normal random distribution based on the layer dimensions.
    std_dev = np.sqrt(2.0 / previous_dim)
    # Initialize weights based on dimensions of current and previous layer
    weights = np.random.normal(loc = 0.0, scale = std_dev, size = (current_dim, previous_dim))

    # Return initialized matrix and seed
    return weights, seed
