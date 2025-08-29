from mnist import MNIST

import neural_network as nn
import hyperparams as hp
import parameter_stepping as step

from network_interface import *


def main():
    # Initialize training dataset
    hp.RAW_DATA = MNIST('mnist_ds')
    images, labels = hp.RAW_DATA.load_training()

    # Initialize training dataset
    test_images, test_labels = hp.RAW_DATA.load_testing()

    # Instantiate all layers
    L1_dimension = len(hp.RAW_DATA.process_images_to_lists(images[0]))

    neural_net = (nn.Layer(L1_dimension), nn.Layer(16), nn.Layer(16), nn.Layer(10))
    L1, L2, L3, L4 = neural_net
    gradients = {L4: step.EWMA(), L3: step.EWMA(), L2: step.EWMA()}

    lowest_cost = 999999
    patience_counter =  0
    while True:
        # Train the neural network
        train_nn(gradients, neural_net, images, labels)

        # Validate neural network
        cost = validate_nn(neural_net, test_images, test_labels)

        print(cost)
        if cost < lowest_cost:
            lowest_cost = cost
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= hp.PATIENCE:
            break

        # Shuffle before next epoch for SGD
        # combined_array = list(zip(images, labels))
        # np.random.shuffle(combined_array)
        # images, labels = zip(*combined_array)



if __name__ == "__main__":
    main()