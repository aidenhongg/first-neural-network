import numpy as np

import neural_network as nn
import backpropagation as bp
import parameter_stepping as step
import hyperparams as hp


def train_nn(gradients, neural_net : tuple[nn.Layer, nn.Layer, nn.Layer, nn.Layer],
             images : list, labels : list):
    training_size = len(images)
    L1, L2, L3, L4 = neural_net

    for x in range(int(training_size / hp.BATCH_SIZE) + 1):
        # Train the neural network on a single batch
        run_batch(x, False, neural_net, images, labels, 0, gradients)

        # Final step of each batch
        for layer, gradient in gradients.items():
            dW, db = gradient.get_gradient()
            layer.update_params(dW, db, hp.STEP_SIZE)


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
        batch_range = range(current_batch_index * hp.BATCH_SIZE,
                            (current_batch_index + 1) * hp.BATCH_SIZE)

    correct_count = 0
    L1, L2, L3, L4 = neural_net
    for i in batch_range:
        if i >= dataset_size:
            break

        label = labels[i]
        encoded_label = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)
        nn.update_label(encoded_label)

        image = images[i]
        l1_neurons = hp.RAW_DATA.process_images_to_lists(image)
        L1.update_neurons(l1_neurons)

        # Forward feed
        for layer in nn.LAYERS[1:]:
            layer.compute_neurons()

        output_neurons = L4.neurons

        # Bulky statement for testing - please revise
        if is_validating:
            correct_count += int(output_neurons.flatten().tolist().index(max(output_neurons.flatten().tolist())) == label)
        total_loss += nn.get_CCE(output_neurons)

        # Backpropagation
        if not is_validating:
            for layer, gradient in gradients.items():
                dW, db = bp.backpropagate(layer)
                gradient.add_gradient(dW, db)
    # Also please revise
    if is_validating:
        print(correct_count / dataset_size)
    return total_loss
