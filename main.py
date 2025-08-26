import numpy as np
from mnist import MNIST

def find_loss(label : int, output_neurons : np.array) -> int:
    """
    Ideal values are as follows (top - down): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """



def He_initialization(current_dim, previous_dim):
    std_dev = np.sqrt(2.0 / previous_dim)
    weights = np.random.normal(loc = 0.0, scale = std_dev, size = (current_dim, previous_dim))
    return weights

def ReLU(weighted_sum : np.ndarray) -> np.ndarray:
    activated_sum = np.copy(weighted_sum)
    activated_sum[activated_sum < 0] = 0
    return activated_sum

def softmax(weighted_sum : np.ndarray) -> np.ndarray:
    stabilized_sum = weighted_sum - np.max(weighted_sum)
    stabilized_exp = np.exp(stabilized_sum)
    activated_sum = stabilized_exp / np.sum(stabilized_exp)
    return activated_sum

class Layer:
    Layers = []
    def __init__(self, dim : int):
        self.dim = dim

        # Create neurons and biases matrices
        self.neurons = np.zeros((dim, 1))
        self.biases = np.zeros((dim, 1))

        # Create weights matrix to calculate this layer's neurons based on previous layer's dim
        if Layer.Layers:
            self.previous_layer = Layer.Layers[-1]
            previous_dim = self.previous_layer.dim
            self.weights = He_initialization(dim, previous_dim)

        # If this is L1 then there will be no weights matrix
        else:
            self.previous_layer, self.weights = None, None

        # Append to layer sequence (to be able to reference previous layers)
        Layer.Layers.append(self)

    def compute_neurons(self):
        # L1 should be from dataset and never computed
        if not self.previous_layer:
            raise Exception("Cannot compute L1 neurons.")

        # Multiply weights and input neurons
        weighted_sum = (self.weights @ self.previous_layer.neurons) + self.biases
        # Apply ReLU for all layers except last - last uses Softmax
        if self == Layer.Layers[-1]:
            activated_sum = softmax(weighted_sum)
        else:
            activated_sum = ReLU(weighted_sum)
        # Update this layers' neurons
        self.update_neurons(activated_sum)

    def update_neurons(self, neurons : list | np.ndarray):
        # Make n x 1 column vector from list. Reject if the dimension is incorrect
        if not isinstance(neurons, np.ndarray):
            neurons = np.array(neurons).reshape(-1, 1)
        if neurons.shape[0] != self.dim:
            neurons = neurons.reshape(-1, 1)
        if neurons.shape != (self.dim, 1):
            raise Exception("Incorrect dimension.")

        self.neurons = neurons

def main():
    # Initialize training dataset
    raw_data = MNIST('mnist_ds')
    images, labels = raw_data.load_training()
    image_count = len(images)

    # Instantiate the first layer
    l1_neurons = raw_data.process_images_to_lists(images[0])
    L1_dimension = len(l1_neurons)
    L1 = Layer(L1_dimension)

    # Instantiate all other layers
    L2 = Layer(16)
    L3 = Layer(16)
    L4 = Layer(10)
    neural_network = [L2, L3, L4]

    total_cost = 0
    for i in range(image_count):
        image = images[i]
        label = labels[i]
        l1_neurons = raw_data.process_images_to_lists(image)


        L1.update_neurons(l1_neurons)
        for layer in neural_network:
            layer.compute_neurons()


if __name__ == "__main__":
    main()