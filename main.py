from mnist import MNIST
from nn_layers import *

def find_loss(label : int, output_neurons : np.array) -> int:
    """
    Ideal values are as follows (top - down): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """


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