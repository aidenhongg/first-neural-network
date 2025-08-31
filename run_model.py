from mnist import MNIST

from epoch import *
import backpropagation as bp
import neural_network

best_parameters = {}

def run(images, labels, test_images, test_labels, for_interactive = False):
    # Clear previous models
    neural_network.LAYERS.clear()

    # Instantiate all layers
    L1_dimension = len(hp.RAW_DATA.process_images_to_lists(images[0]))
    neural_net = (nn.Layer(L1_dimension), nn.Layer(16), nn.Layer(16), nn.Layer(10))
    L1, L2, L3, L4 = neural_net

    # Load previous model if running interactively
    if hp.LOAD_MODEL:
        for index, layer in enumerate(neural_net[1:]):
            weight = np.load(f"./IO/output/model/L{index + 2}/weight.npy")
            bias = np.load(f"./IO/output/model/L{index + 2}/bias.npy")

            layer.weights = weight
            layer.bias = bias

        if for_interactive:
            return

    gradients = {L4: bp.EWMA(), L3: bp.EWMA(), L2: bp.EWMA()}
    lowest_cost = 999999
    patience_counter = 0

    while True:
        # Train the neural network
        train_nn(gradients, images, labels)

        # Validate neural network
        accuracy, cost = validate_nn(test_images, test_labels)
        print(f"Epoch analyzed: {accuracy} accuracy, {cost} cost")
        # Buffer to only track meaningful improvements
        if cost < lowest_cost - hp.PATIENCE_BUFFER:
            lowest_cost = cost
            patience_counter = 0
            if hp.SAVE_MODEL:
                for layer in neural_network.LAYERS[1:]:
                    best_parameters[layer] = [layer.weights.copy(), layer.biases.copy()]
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
    # Initialize training dataset
    hp.RAW_DATA = MNIST('./IO/mnist_ds')
    images, labels = hp.RAW_DATA.load_training()

    # Initialize testing dataset
    test_images, test_labels = hp.RAW_DATA.load_testing()
    accuracy, lowest_cost, seed = run(images, labels, test_images, test_labels)

    if hp.SAVE_MODEL:
        with open("./IO/output/output.txt", 'w') as file:
            file.write(f"Accuracy: {accuracy}, Lowest cost: {lowest_cost}, Seed {seed}")

        for index, parameters in enumerate(best_parameters.values()):
            weight = parameters[0]
            bias = parameters[1]
            np.save(f"./IO/output/model/L{index + 2}/weight.npy", weight)
            np.save(f"./IO/output/model/L{index + 2}/bias.npy", bias)
