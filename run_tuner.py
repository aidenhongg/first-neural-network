import csv
from mnist import MNIST
from itertools import product

import hyperparams as hp
import run_model

def main():
    # MOMENTUM
    beta1 = [0.85, 0.87, 0.89, 0.91, 0.93, 0.95]

    # VARIANCE
    beta2 = [0.991, 0.993, 0.995, 0.997, 0.999]

    # LEARNING_RATE
    learning_rate = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]

    combinations = list(product(beta1, beta2, learning_rate))

    """
    Format of cost_by_hyperparameter.csv:
    
    columns = ["Momentum (0.85 - 0.95, 0.02)",
                                 "Variance (0.991 - 0.999, 0.002)",
                                 "Learning Rate (0.0001 - 0.0009, 0.0002)",
                                 "Seed",
                                 "Lowest cost",
                                 "Highest Accuracy"]
    df = pd.DataFrame(columns = columns)
    df.to_csv('./output/cost_by_hyperparameter.csv', index = False)
    """

    # Initialize training dataset
    hp.RAW_DATA = MNIST('./IO/mnist_ds')
    images, labels = hp.RAW_DATA.load_training()

    # Initialize training dataset
    test_images, test_labels = hp.RAW_DATA.load_testing()
    run_model.run(images, labels, test_images, test_labels)

    for row in combinations:
        hp.MOMENTUM = row[0]
        hp.VARIANCE = row[1]
        hp.LEARNING_RATE = row[2]
        lowest_cost = 1000
        best_accuracy = 0
        best_seed = 0

        for _ in range(10):
            while True:
                try:
                    accuracy, cost, seed = run_model.run(images, labels, test_images, test_labels)
                    if accuracy > 0.12:
                        break
                except Exception as e:
                    print(e)
            print("Model trial success")

            if cost < lowest_cost:
                lowest_cost = cost
                best_accuracy = accuracy
                best_seed = seed
        new_row = (hp.MOMENTUM, hp.VARIANCE, hp.LEARNING_RATE,
                   best_seed, lowest_cost, best_accuracy)
        print("10 seeds trialed - entry recorded")

        with open('./IO/output/cost_by_hyperparameter.csv', 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

if __name__ == "__main__":
    main()