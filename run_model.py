from mnist import MNIST

from epoch import *
import backpropagation as bp

"""
run_model.py creates and trains a new model.
    -Applies the PATIENCE and PATIENCE_BUFFER hyperparameters between epochs
    -Responsible for loading the saved model if LOAD_MODEL = True
    -Saves the model, the SEED used to train it, and its stats if SAVE_MODEL = True """

# Container to hold optimized parameters before saving them to a persistent file
best_parameters = {}

""""
run_model.run() controls all model instantiation and training operations.
    
1. Instantiates neural network object
2. Loads saved model states
3. Trains the model using a mainloop
    On each iteration:
    -Trains the model on an epoch of the training data
    -Validates the model using the testing data and the CCE cost function
    -Handles early stop using PATIENCE and PATIENCE_BUFFER hyperparameters
    specified in hyperparameters_flags.py
        
Returns the raw accuracy on the testing set, the average cost on the testing set, and the
seed used when randomly initializing the network parameters, in that order. """""

def run(for_interactive = False) -> tuple[float, float, int]:
    # Load training and testing dataset
    images, labels = hp.RAW_DATA.load_training()
    test_images, test_labels = hp.RAW_DATA.load_testing()

    # Clear previous network parameters if they persist
    nn.LAYERS.clear()

    # Instantiate all layers - 2 hidden layers of 16 neurons
    L1_dimension = len(hp.RAW_DATA.process_images_to_lists(images[0]))
    L1, L2, L3, L4 = nn.Layer(L1_dimension), nn.Layer(16), nn.Layer(16), nn.Layer(10)

    # Also useful to have a pointer to all layers but input layers -
    # for saving and loading parameters
    hidden_and_output_layers = nn.LAYERS[1:]

    # Load saved parameters to neural network object if LOAD_MODEL = True
    if hp.LOAD_MODEL:
        # L{index + 2} refers to the L2, L3, L4 folders.
        for index, layer in enumerate(hidden_and_output_layers):
            weight = np.load(f"_IO/output/model/L{index + 2}/weight.npy")
            bias = np.load(f"_IO/output/model/L{index + 2}/bias.npy")

            # Set layer parameters (weight and bias) to saved parameters
            layer.weights = weight
            layer.bias = bias

        # End function early and do not train model if running interactively
        if for_interactive:
            return

    # The 'gradients' dictionary attaches Exponential Weighted Moving Averages to each layer -
    # helping calculate gradient shifts on each parameter between each batch and epoch.
    gradients = {L4: bp.EWMA(), L3: bp.EWMA(), L2: bp.EWMA()}
    # It is instantiated backwards for backpropagation purposes.

    # lowest_cost tracks the lowest cost after each epoch
    lowest_cost = np.inf

    # patience_counter tracks the number of epochs that fail to deliver a meaningfully lower cost
    # - Add 1 if (cost < (lowest_cose - hp.PATIENCE_BUFFER))

    # If the model fails to deliver a low enough cost for enough epochs, do an early stop
    # - if patience_counter > hp.PATIENCE, stop training loop
    patience_counter = 0

    # Training loop
    while True:
        # Train the neural network
        train_nn(gradients, images, labels)

        # Validate neural network
        accuracy, cost = validate_nn(test_images, test_labels)
        # Output status message
        print(f"Epoch analyzed: {accuracy} accuracy, {cost} cost")

        # Detect if a meaningful cost reduction was delivered
        if cost < lowest_cost - hp.PATIENCE_BUFFER:

            # Track it with lowest_cost and reinitialize the patience_counter
            lowest_cost = cost
            patience_counter = 0

            # If SAVE_MODEL = True, update best_parameters with the current parameters
            # to save them after runtime
            if hp.SAVE_MODEL:
                for layer in hidden_and_output_layers:
                    best_parameters[layer] = [layer.weights.copy(), layer.biases.copy()]

        # If there was NO meaningful cost reduction, increment the patience_counter
        else:
            patience_counter += 1

        # If there's NO meaningful cost reduction for too long, do early stop
        if patience_counter >= hp.PATIENCE:
            break

        # Shuffle the data between epochs to randomize batches
        shuffled_labels = list(zip(images, labels))
        np.random.shuffle(shuffled_labels)
        images, labels = zip(*shuffled_labels)

    # Return the seed as well so this training session can be reproduced later
    return accuracy, lowest_cost, hp.SEED

if __name__ == "__main__":
    # Load the dataset into project-wide variable
    hp.RAW_DATA = MNIST('_IO/mnist_ds')
    # Train model until sufficiently optimal, then store its stats
    accuracy, lowest_cost, seed = run()

    # Save the model if SAVE_MODEL flag = True
    if hp.SAVE_MODEL:
        # Output the performance statistics of the model to statistics.txt
        with open("_IO/output/model/statistics.txt", 'w') as file:
            file.write(f"Accuracy: {accuracy}, Lowest cost: {lowest_cost}, Seed: {seed}")

        # Store the parameters - weights and biases - to weight.npy and bias.npy
        # L2, L3, and L4 folders store parameters of each layer
        for index, parameters in enumerate(best_parameters.values()):
            weight = parameters[0]
            bias = parameters[1]
            np.save(f"_IO/output/model/L{index + 2}/weight.npy", weight)
            np.save(f"_IO/output/model/L{index + 2}/bias.npy", bias)

        # Output status message
        print("Model saved.")
