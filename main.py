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
    patience_counter = 0
    while True:
        # Train the neural network
        train_nn(gradients, neural_net, images, labels)

        # Validate neural network
        accuracy, cost = validate_nn(neural_net, test_images, test_labels)

        print(accuracy)
        print(cost)
        # Buffer to only track meaningful improvements
        if cost < lowest_cost - 0.001:
            lowest_cost = cost
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= hp.PATIENCE:
            break

        # Shuffle before next epoch for SGD
        shuffled_labels = list(zip(images, labels))
        np.random.shuffle(shuffled_labels)
        images, labels = zip(*shuffled_labels)


    return accuracy, lowest_cost, hp.SEED



if __name__ == "__main__":
    main()