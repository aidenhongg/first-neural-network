import mnist

import neural_network as nn
import backpropagation as bp
import parameter_stepping as step
import label as lb

from nn_functions import *
from mnist import MNIST

# Batch size between each gradient adjustment
BATCH_SIZE : int = 32

# Step size
STEP_SIZE : float = 0.1

# Dataset
RAW_DATA : mnist.MNIST = mnist.MNIST()

# Patience
PATIENCE : int = 5

def train_nn(neural_net : tuple[nn.Layer, nn.Layer, nn.Layer, nn.Layer],
             images : list, labels : list):
    training_size = len(images)
    L1, L2, L3, L4 = neural_net

    for x in range(int(training_size / BATCH_SIZE) + 1):
        gradients = {L4 : step.EWMA(), L3 : step.EWMA(), L2 : step.EWMA()}

        # Train the neural network on a single batch
        run_batch(x, False, neural_net, images, labels, 0, gradients)

        # Final step of each batch
        for layer, gradient in gradients.items():
            dW, db = gradient.get_gradient()
            layer.update_params(dW, db, STEP_SIZE)

        # After each batch
        step.EWMA.clear_instances()

def validate_nn(neural_net : tuple[nn.Layer, nn.Layer, nn.Layer, nn.Layer],
             images : list, labels : list):
    testing_size = len(images)
    total_loss = run_batch(0, True, neural_net, images,
                           labels, 0)
    cost = get_cost(total_loss, testing_size)
    print(cost)


def run_batch(current_batch_index : int, is_validating : bool,
              neural_net, images : list, labels : list, total_loss, gradients = None):

    dataset_size = len(images)
    if is_validating:
        batch_range = range(dataset_size)
    else:
        batch_range = range(current_batch_index * BATCH_SIZE,
                            (current_batch_index + 1) * BATCH_SIZE)

    L1, L2, L3, L4 = neural_net
    for i in batch_range:
        if i >= dataset_size:
            break

        label = labels[i]
        encoded_label = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)
        lb.update_label(encoded_label)

        image = images[i]
        l1_neurons = RAW_DATA.process_images_to_lists(image)
        L1.update_neurons(l1_neurons)

        # Forward feed
        for layer in nn.LAYERS[1:]:
            layer.compute_neurons()

        output_neurons = L4.neurons
        total_loss += get_CCE(output_neurons)

        # Backpropagation
        if not is_validating:
            for layer, gradient in gradients.items():
                dW, db = bp.backpropagate(layer)
                gradient.add_gradient(dW, db)

    return total_loss


def main():
    # Initialize training dataset
    global RAW_DATA
    RAW_DATA = MNIST('mnist_ds')
    images, labels = RAW_DATA.load_training()
    training_size = len(images)

    # Instantiate all layers
    L1_dimension = len(RAW_DATA.process_images_to_lists(images[0]))
    neural_net = (nn.Layer(L1_dimension), nn.Layer(16), nn.Layer(16), nn.Layer(10))

    # Train the neural network
    train_nn(neural_net, images, labels)

    # Validate neural network
    test_images, test_labels = RAW_DATA.load_testing()
    validate_nn(neural_net, test_images, test_labels)



if __name__ == "__main__":
    main()