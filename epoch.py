import numpy as np

import neural_network as nn
import backpropagation as bp
import hyperparams as hp

def train_nn(gradients, images : list, labels : list):

    dataset_size = len(images)

    for x in range(int(dataset_size / hp.BATCH_SIZE) + 1):

        batch_range = range(x * hp.BATCH_SIZE,
                            (x + 1) * hp.BATCH_SIZE)

        examples_analyzed = 0
        batch_averages = {}

        for layers in gradients.keys():
            batch_averages[layers] = ["dW", "db"]

        # Train the neural network on a single batch
        for i in batch_range:
            if i >= dataset_size:
                break
            image = images[i]
            label = labels[i]
            run_example(image, label)

            # Backpropagation
            for layer in gradients.keys():
                dW, db = bp.backpropagate(layer)
                if examples_analyzed == 0:
                    batch_averages[layer] = [dW, db]

                else:
                    batch_averages[layer][0] += dW
                    batch_averages[layer][1] += db

            examples_analyzed += 1

        # Final step after each batch
        for layer, gradient in gradients.items():
            if examples_analyzed > 0:
                dW_average, db_average = batch_averages[layer]

                dW_average = dW_average / examples_analyzed
                db_average = db_average / examples_analyzed

                gradient.add_gradient(dW_average, db_average)

            dW, db = gradient.get_gradient()
            layer.update_params(dW, db, hp.LEARNING_RATE)

def validate_nn(images : list, labels : list):

    correct_count = 0
    total_loss = 0

    testing_size = len(images)
    batch_range = range(testing_size)

    for i in batch_range:
        image = images[i]
        label = labels[i]
        output_neurons, loss = run_example(image, label)

        total_loss += loss

        # See if the selection was correct for raw accuracy
        selection = selection_from_output(output_neurons)
        correct_count += int(selection == label)

    cost = nn.get_cost(total_loss, testing_size)
    accuracy = correct_count / testing_size
    return accuracy, cost


def run_example(image : list, label : list):
    L1, L2, L3, L4 = nn.LAYERS

    encoded_label = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)
    nn.update_label(encoded_label)

    l1_neurons = hp.RAW_DATA.process_images_to_lists(image)
    L1.update_neurons(l1_neurons)

    # Forward feed
    for layer in nn.LAYERS[1:]:
        layer.compute_neurons()

    output_neurons = L4.neurons
    loss = nn.get_CCE(output_neurons)

    return output_neurons, loss

def selection_from_output(neurons : np.ndarray):
    output_list = neurons.flatten().tolist()
    max_value = max(output_list)
    selection = output_list.index(max_value)

    return selection
