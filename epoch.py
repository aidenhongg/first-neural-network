import numpy as np

import neural_network as nn
import backpropagation as bp
import hyperparameters_flags as hp

"""
epoch.py handles feeding the NN an epoch of the dataset while updating
the parameter gradients of the network after each batch.
- Handles training and validation 
- Manages feed-forward and backpropagation
"""

def train_nn(gradients : dict[nn.Layer : bp.EWMA],
             images : list, labels : list):
    """
    train_nn() passes an epoch of the data to the current neural network and updates its
    parameters. 
        1. Maintains an average gradient over a batch (the batch average gradient) 
        2. Passes each average to its respective EWMA for its parameter, 
        3. Uses the EWMA to calculate the appropriate adjustment for each parameter,
        4. Updates all the parameters.
    Batch size is specified in hp.BATCH_SIZE """""

    # Get the size of the epoch to pass
    dataset_size = len(images)

    # Find the number of batches to pass and iterate through that many batches
    for batches_passed in range(int(dataset_size / hp.BATCH_SIZE) + 1):

        # Get the indices of all examples for the batch within the total dataset
        batch_range = range(batches_passed * hp.BATCH_SIZE,
                            (batches_passed + 1) * hp.BATCH_SIZE)

        # Counter to track how many examples were successfully analyzed
        examples_analyzed = 0

        # Container to hold gradient averages across each batch
        # Format is: {Layer pointer : [Weight gradient average, bias gradient average, ...]}
        batch_averages = {}
        for layers in gradients.keys():
            batch_averages[layers] = ["dW", "db"]
            # Gradients holds layers backwards - so will batch_averages.

        # Pass a single batch - iterate through all batch indices
        for example_index in batch_range:
            # If our index exceeds the epoch size, break early
            # For cases where the batch size doesn't evenly divide the epoch size
            if example_index >= dataset_size:
                break

            # Get example image and label
            image = images[example_index]
            label = labels[example_index]

            # Feed forward the example through the network
            # Updates their neurons - necessary for backpropagation
            feed_forward(image, label)

            # Backpropagate through each layer from L4 - L2
            for layer in gradients.keys():
                # Get the weight and bias gradients for the current layer
                dW, db = bp.backpropagate(layer)

                # If no examples have been analyzed yet, instantiate the average batch
                # gradient for that layer as the gradient itself
                if examples_analyzed == 0:
                    batch_averages[layer] = [dW, db]

                # Otherwise add the gradient to the current batch average for that layer
                else:
                    batch_averages[layer][0] += dW
                    batch_averages[layer][1] += db

            # Iterate the examples_analyzed counter
            examples_analyzed += 1

        # After each batch - update each parameter with the appropriate gradient
        for layer, gradient in gradients.items():

            # Ensure that some examples were actually analyzed
            if examples_analyzed > 0:

                # Calculate the average for all the gradients given in the batch
                # for each parameter of the current layer - weight and bias
                dW_average, db_average = batch_averages[layer]
                dW_average = dW_average / examples_analyzed
                db_average = db_average / examples_analyzed

                # Add this average gradient to the EWMA of each parameter
                gradient.add_gradient(dW_average, db_average)

            # Get the final gradient to be applied from the EWMA of each parameter
            dW, db = gradient.get_gradient()

            # Finish analyzing the batch by updating every parameter with these final gradients.
            layer.update_params(dW, db)

def validate_nn(images : list, labels : list) -> tuple[float, float]:
    """
    train_nn() passes the entire testing dataset to the current neural network. It tracks
    and returns:
        - The average cost of all the testing examples
        - The raw performance accuracy of the model (% correct). """""

    # correct_count tracks the number of examples the model predicted correctly
    correct_count = 0
    # total_loss tracks the total loss across the testing epoch
    total_loss = 0

    # Get the size of the validation set
    testing_size = len(images)

    # Iterate through the validation set
    for index in range(testing_size):
        # Get the image and label from the example index
        image = images[index]
        label = labels[index]

        # Feed forward the example and get the output and loss
        output_neurons, loss = feed_forward(image, label)

        # Add the loss for the example to the total loss
        total_loss += loss

        # See if the prediction was correct
        selection = selection_from_output(output_neurons)
        # If it was, increment the correct_count tracker
        correct_count += int(selection == label)

    # Get the average cost across the testing set, the raw accuracy, and return
    cost = nn.get_cost(total_loss, testing_size)
    accuracy = correct_count / testing_size
    return accuracy, cost


def feed_forward(image : list, label : list) -> tuple[np.ndarray, float]:
    """
    Feeds the given example through the neural network, updating its neurons.
    - Returns the resulting prediction and loss
    """""
    # Unpack the first and last layers
    L1, L4 = nn.LAYERS[0], nn.LAYERS[-1]

    # Get the one hot-encoded label - the ideal output vector
    encoded_label = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)
    # Update the neural network for loss calculation
    nn.update_label(encoded_label)

    # Update the input neurons on the first layer with the input vector
    l1_neurons = hp.RAW_DATA.process_images_to_lists(image)
    L1.update_neurons(l1_neurons)

    # Compute the resulting neurons of each layer in order
    for layer in nn.LAYERS[1:]:
        layer.compute_neurons()

    # Get the output neurons
    output_neurons = L4.neurons
    # Calculate the loss
    loss = nn.get_CCE(output_neurons)

    # Return the output and the loss
    return output_neurons, loss

def selection_from_output(neurons : np.ndarray) -> int:
    """
    Calculates the model's prediction from its output vector. """""

    neuron_total = neurons.sum()

    output_tracker = 0
    max_value = 0
    model_prediction = None
    for index, value in enumerate(neurons):
        output_tracker += value

        if value > max_value:
            max_value = value
            model_prediction = index

        if max_value > (neuron_total - output_tracker):
            break

    return model_prediction
