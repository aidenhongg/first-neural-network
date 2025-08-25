import numpy as np
from mnist import MNIST

def RelU(x : int) -> int:
    return max(0, x)

class Layer:
    Layers = []
    def __init__(self, layer_dim : int):
        self.layer_dim = layer_dim

        # Create neurons and biases matrices
        self.neurons = np.zeros((layer_dim, 1))
        self.biases = np.zeros((layer_dim, 1))

        # Create weights matrix to calculate this layer's neurons based on previous layer's dim
        if Layer.Layers:
            previous_layer_dim = Layer.Layers[-1].layer_dim
            self.weights = np.zeros((previous_layer_dim, layer_dim))

        # If this is L1 then there will be no weights matrix
        else:
            self.weights = None

        # Append to layer sequence (to be able to reference previous layers)
        Layer.Layers.append(self)

    def random_weights_biases(self):
        pass

    def compute_neurons(self):
        pass

    def update_neurons(self, neurons : list | np.ndarray):
        # Make n x 1 column vector from list. Reject if the dimension is incorrect
        if not isinstance(neurons, np.ndarray):
            neurons = np.array(neurons).reshape(-1, 1)
        if neurons.shape[0] != self.layer_dim:
            neurons = neurons.reshape(-1, 1)
        if neurons.shape != (self.layer_dim, 1):
            raise Exception("Incorrect dimension.")

        self.neurons = neurons

def main():
    # Initialize training dataset
    raw_data = MNIST('mnist_ds')
    images, labels = raw_data.load_training()

    # Instantiate the first layer
    l1_neurons = raw_data.process_images_to_lists(images[0])
    L1_dimension = len(l1_neurons)
    L1 = Layer(L1_dimension)

    # Instantiate all other layers
    L2 = Layer(16)
    L3 = Layer(16)
    L4 = Layer(10)

    L1.update_neurons(l1_neurons)
    print(L1.neurons)


if __name__ == "__main__":
    main()