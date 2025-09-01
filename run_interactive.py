import random
from mnist import MNIST

import epoch
import run_model
import hyperparameters_flags as hp
import neural_network.neural_network as nn

""""
Starts an interactive mode where users can interact with an already-trained model.

    -A model must be stored in ./IO/output/model. This can be obtained by training and
    storing a model with run_model.py.
    
    -An image of a handwritten drawing of a digit is presented through the console.
    Users must then predict what digit it is. The image data is then fed through the model
    to get the model's prediction. Finally, the correct label is shown, either confirming or
    refuting the predictions. """""

def main():
    """"
    The interactive experience operates through a loop that continues getting user input.
    The images and labels be shown are randomly selected from the testing dataset. """""
    # Set LOAD_MODEL flag to True if not already
    hp.LOAD_MODEL = True

    # Load dataset to project-wide variable
    hp.RAW_DATA = MNIST('_IO/mnist_ds')
    # Initialize saved model state in ./_IO/output/model as runtime model
    run_model.run(True)

    # Get and print performance statistics of the loaded model
    with open("_IO/output/model/statistics.txt", "r") as file:
        stats = file.readline()
    print(f"Loaded model performance: {stats}")

    # Pointer to first layer will be useful as it must regularly be updated with the input image data
    first_layer = nn.LAYERS[0]

    # A pointer to the last layer is also needed as it holds the model's prediction
    last_layer = nn.LAYERS[-1]

    # Load images and labels
    test_images, test_labels = hp.RAW_DATA.load_testing()

    # Inform user of possible inputs and enter input loop
    print("Press enter for interactive mode. Submit 'exit' to exit.")
    while True:
        # Get input - if they only pressed enter, it will be ""
        user_input = input()

        # "" is a False boolean so this triggers if they only pressed enter
        if not user_input:
            # Select random example and get image and label
            selected_index = random.randint(0, len(test_images) - 1)
            image = test_images[selected_index]
            label = test_labels[selected_index]

            # Print the image to console
            print(hp.RAW_DATA.display(image))

            # Get the user's prediction - must be an integer
            user_prediction_input = input("Guess the number: ")
            while not user_prediction_input.isdigit():
                user_prediction_input = input("Guess a valid integer to proceed: ")

            user_prediction = int(user_prediction_input)

            # Update the first layer (input layer)'s neurons with the image data
            l1_neurons = hp.RAW_DATA.process_images_to_lists(image)
            first_layer.update_neurons(l1_neurons)

            # Feed forward the model and get the prediction
            for layer in nn.LAYERS[1:]:
                layer.compute_neurons()
            output_neurons = last_layer.neurons

            # Translate the output neurons to a selection
            # Whichever index has the highest value will be the model's prediction
            model_prediction = epoch.selection_from_output(output_neurons)

            # Show user the model's guess
            print(f"Model guessed: {model_prediction}.")

            # Different console outputs depending on whose prediction was correct
            if model_prediction == user_prediction == label:
                print("You both got it right.")

            elif user_prediction == label and model_prediction != label:
                print("The model turned out to be wrong.")

            elif user_prediction != label and model_prediction == label:
                print("Looks like the model is smarter than you.")

        # Break loop and jump to program end if user enters 'exit'
        elif user_input == "exit":
            break

        # For all invalid inputs display valid inputs and restart loop
        else:
            print("Press enter for interactive mode. Submit 'exit' to exit.")

if __name__ == "__main__":
    main()