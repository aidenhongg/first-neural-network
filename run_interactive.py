import random
from mnist import MNIST

import epoch
import run_model
import hyperparams as hp
import neural_network.neural_network as nn


def main():
    hp.LOAD_MODEL = True
    hp.RAW_DATA = MNIST('./IO/mnist_ds')

    test_images, test_labels = hp.RAW_DATA.load_testing()
    run_model.run(test_images, None, None, None, True)

    with open("./IO/output/output.txt", "r") as file:
        stats = file.readline()
    print(f"Loaded model performance: {stats}")

    first_layer = nn.LAYERS[0]
    last_layer = nn.LAYERS[-1]

    print("Press enter for interactive mode. Submit 'exit' to exit.")
    while True:
        user_input = input()
        if not user_input:
            selected_index = random.randint(0, len(test_images) - 1)
            image = test_images[selected_index]
            label = test_labels[selected_index]

            print(hp.RAW_DATA.display(image))

            guess_input = input("Guess the number: ")
            while not guess_input.isdigit():
                guess_input = input("Guess a valid integer to proceed: ")

            human_guess = int(guess_input)

            l1_neurons = hp.RAW_DATA.process_images_to_lists(image)
            first_layer.update_neurons(l1_neurons)

            # Forward feed
            for layer in nn.LAYERS[1:]:
                layer.compute_neurons()
            output_neurons = last_layer.neurons

            machine_guess = epoch.selection_from_output(output_neurons)

            print(f"Model guessed: {machine_guess}.")

            if machine_guess == human_guess == label:
                print("You both got it right.")

            elif human_guess == label and machine_guess != label:
                print("The model turned out to be wrong.")

            elif human_guess != label and machine_guess == label:
                print("Looks like the model is smarter than you.")

        elif user_input == "exit":
            break

        else:
            print("Press enter for interactive mode. Submit 'exit' to exit.")

if __name__ == "__main__":
    main()