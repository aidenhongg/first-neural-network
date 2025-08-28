from mnist import MNIST

import numpy as np

import neural_network as nn
import backpropagation as bp
import parameter_stepping as step


# Batch size between each gradient adjustment
BATCH_SIZE : int = 32

# Step size
STEP_SIZE : float = 0.0005

# Dataset
RAW_DATA : MNIST = MNIST()

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
    cost = nn.get_cost(total_loss, testing_size)
    return cost


def run_batch(current_batch_index : int, is_validating : bool,
              neural_net, images : list, labels : list, total_loss, gradients = None):

    dataset_size = len(images)
    if is_validating:
        batch_range = range(dataset_size)
    else:
        batch_range = range(current_batch_index * BATCH_SIZE,
                            (current_batch_index + 1) * BATCH_SIZE)

    correct_count = 0
    L1, L2, L3, L4 = neural_net
    for i in batch_range:
        if i >= dataset_size:
            break

        label = labels[i]
        encoded_label = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)
        nn.update_label(encoded_label)

        image = images[i]
        l1_neurons = RAW_DATA.process_images_to_lists(image)
        L1.update_neurons(l1_neurons)

        # Forward feed
        for layer in nn.LAYERS[1:]:
            layer.compute_neurons()

        output_neurons = L4.neurons
        if is_validating:
            correct_count += int(output_neurons.flatten().tolist().index(max(output_neurons.flatten().tolist())) == label)
        total_loss += nn.get_CCE(output_neurons)

        # Backpropagation
        if not is_validating:
            for layer, gradient in gradients.items():
                dW, db = bp.backpropagate(layer)
                gradient.add_gradient(dW, db)
    if is_validating:
        print(correct_count / dataset_size)
    return total_loss


def main():
    # Initialize training dataset
    global RAW_DATA
    RAW_DATA = MNIST('mnist_ds')
    images, labels = RAW_DATA.load_training()

    # Instantiate all layers
    L1_dimension = len(RAW_DATA.process_images_to_lists(images[0]))
    neural_net = (nn.Layer(L1_dimension), nn.Layer(16), nn.Layer(16), nn.Layer(10))

    lowest_cost = np.inf
    patience_counter =  0
    while True:
        # Train the neural network
        train_nn(neural_net, images, labels)

        # Validate neural network
        test_images, test_labels = RAW_DATA.load_testing()
        cost = validate_nn(neural_net, test_images, test_labels)

        print(cost)
        if cost < lowest_cost:
            lowest_cost = cost
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break


if __name__ == "__main__":
    main()