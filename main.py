import neural_network as nn
import backpropagation as bp
import parameter_stepping as step

from nn_functions import *
from mnist import MNIST

# Batch size between each gradient adjustment
BATCH_SIZE : int = 32

# Step size
STEP_SIZE : float = 0.1

def main():
    # Initialize training dataset
    raw_data = MNIST('mnist_ds')
    images, labels = raw_data.load_training()
    training_size = len(images)

    # Instantiate the first layer
    l1_neurons = raw_data.process_images_to_lists(images[0])
    L1_dimension = len(l1_neurons)

    # Instantiate all layers
    L1, L2, L3, L4 = nn.Layer(L1_dimension), nn.Layer(16), nn.Layer(16), nn.Layer(10)

    for x in range(int(training_size / BATCH_SIZE)):
        gradients = {L4 : step.EWMA(), L3 : step.EWMA(), L2 : step.EWMA()}

        # Train the neural network on a single batch
        for i in range(x * BATCH_SIZE, (x + 1) * BATCH_SIZE):
            label = labels[i]
            encoded_label = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)
            nn.update_label(encoded_label)

            image = images[i]
            l1_neurons = raw_data.process_images_to_lists(image)
            L1.update_neurons(l1_neurons)

            # Forward feed
            for layer in nn.LAYERS[1:]:
                layer.compute_neurons()

            # Backpropagation
            for layer, gradient in gradients.items():
                dW, db = bp.backpropagate(layer)
                gradient.add_gradient(dW, db)

        # Final step of each batch
        for layer, gradient in gradients.items():
            dW, db = gradient.get_gradient()
            layer.update_params(dW, db, STEP_SIZE)

        # After each batch
        step.EWMA.clear_instances()

    # Validation loop after each epoch
    test_images, test_labels = raw_data.load_testing()
    testing_size = len(test_images)

    # Track total loss as each data point is processed
    total_loss = 0
    for i in range(testing_size):
        label = test_labels[i]
        encoded_label = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)
        nn.update_label(encoded_label)

        image = test_images[i]
        l1_neurons = raw_data.process_images_to_lists(image)
        L1.update_neurons(l1_neurons)

        # Forward feed
        for layer in nn.LAYERS[1:]:
            layer.compute_neurons()
        output_neurons = L4.neurons
        total_loss += get_CCE(output_neurons)

    cost = get_cost(total_loss, testing_size)
    print(cost)




if __name__ == "__main__":
    main()