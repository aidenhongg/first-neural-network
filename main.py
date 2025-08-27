import neural_network as nn
import backpropagation as bp
from cost_functions import *
from mnist import MNIST


def main():
    # Initialize training dataset
    raw_data = MNIST('mnist_ds')
    images, labels = raw_data.load_training()
    training_set_size = len(images)

    # Instantiate the first layer
    l1_neurons = raw_data.process_images_to_lists(images[0])
    L1_dimension = len(l1_neurons)

    # Instantiate all layers
    L1, L2, L3, L4 = nn.Layer(L1_dimension), nn.Layer(16), nn.Layer(16), nn.Layer(10)

    # Track total loss as each data point is processed
    total_loss = 0

    # Run the neural network on all training data points
    for i in range(training_set_size):
        label = labels[i]
        encoded_label = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)
        nn.update_label(encoded_label)

        image = images[i]
        l1_neurons = raw_data.process_images_to_lists(image)
        L1.update_neurons(l1_neurons)

        for layer in nn.LAYERS[1:]:
            layer.compute_neurons()
        output_neurons = L4.neurons
        total_loss += get_CCE(output_neurons)

        for layer in nn.LAYERS[::-1]:
            bp.backpropagate(layer)

        break


    cost = get_cost(total_loss, training_set_size)
    print(cost)



if __name__ == "__main__":
    main()