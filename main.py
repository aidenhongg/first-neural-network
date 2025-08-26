from mnist import MNIST
from neural_network import *
from cost_functions import *

def main():
    # Initialize training dataset
    raw_data = MNIST('mnist_ds')
    images, labels = raw_data.load_training()
    training_set_size = len(images)

    # Instantiate the first layer
    l1_neurons = raw_data.process_images_to_lists(images[0])
    L1_dimension = len(l1_neurons)
    L1 = Layer(L1_dimension)

    # Instantiate all other layers
    L2, L3, L4 = Layer(16), Layer(16), Layer(10)

    # Track total loss as each data point is processed
    total_loss = 0

    # Run the neural network on all training data points
    for i in range(training_set_size):
        image = images[i]
        label = labels[i]

        l1_neurons = raw_data.process_images_to_lists(image)
        L1.update_neurons(l1_neurons)

        for layer in LAYERS[1:]:
            layer.compute_neurons()
        output_neurons = L4.neurons

        total_loss += get_CCE(label, output_neurons)

    cost = get_cost(total_loss, training_set_size)
    print(cost)


if __name__ == "__main__":
    main()