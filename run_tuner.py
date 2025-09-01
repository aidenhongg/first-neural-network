import os
import csv
import pandas as pd
from mnist import MNIST
from itertools import product

import run_model
import hyperparameters_flags as hp

"""
Trains models based on different hyperparameters to find their optimal values
    - Hyperparameters being adjusted are momentum (beta1), variance (beta2), and 
    learning rate.
    - 10 different random seeds are instantiated to find the best performance for 
    each different set of hyperparameters.
    - Each set of hyperparameters, with their best accuracy and score, and the seed 
    used to initialize for that best score, is written as a new entry to 
    ./output/cost_by_hyperparameter.csv. 
    
Model training is currently too slow for random hyperparameter optimization.
Some possible solutions are as follows:
    - Move matrix operations to GPU
    - Train multiple models with different hyperparameters in parallel
    - Commit matrix multiplication operations in parallel (EWMA updates) """

def main():
    # Set of MOMENTUM (beta1) values to trial (0.85 - 0.95), step of 0.02
    beta1 = [0.85, 0.87, 0.89, 0.91, 0.93, 0.95]

    # Set of VARIANCE (beta2) values to trial (0.991 - 0.999), step of 0.02
    beta2 = [0.991, 0.993, 0.995, 0.997, 0.999]

    # Set of LEARNING_RATE values to trial (0.0001 - 0.0009), step of 0.0002
    learning_rate = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]

    # List all possible combinations of MOMENTUM, VARIANCE, and LEARNING_RATE
    # values to trial
    combinations = list(product(beta1, beta2, learning_rate))

    csv_filepath = '_IO/output/cost_by_hyperparameter.csv'
    if not os.path.exists(csv_filepath):

        columns = ["Momentum (0.85 - 0.95, 0.02)",
                                 "Variance (0.991 - 0.999, 0.002)",
                                 "Learning Rate (0.0001 - 0.0009, 0.0002)",
                                 "Seed",
                                 "Lowest cost",
                                 "Highest Accuracy"]
        df = pd.DataFrame(columns = columns)
        df.to_csv(csv_filepath, index = False)

    # Load dataset to project-wide variable
    hp.RAW_DATA = MNIST('_IO/mnist_ds')

    # Start iterating through all different sets of hyperparameters
    for row in combinations:
        # Set hyperparameters of model during runtime
        hp.MOMENTUM = row[0]
        hp.VARIANCE = row[1]
        hp.LEARNING_RATE = row[2]

        # Container to track the lowest recorded cost with its correlated
        # accuracy and initialization seed
        lowest_cost = 1000
        best_accuracy = 0
        best_seed = 0

        # Loop through all hyperparameters 10 times to find the best seed
        for _ in range(10):
            while True:
                # Handle any runtime exceptions on a trial instead of ending testing
                try:
                    # Run the model to get its performance
                    accuracy, cost, seed = run_model.run()

                    # Only consider trials where accuracy goes above 12% (a random guess)
                    if accuracy > 0.12:
                        break

                except Exception as e:
                    print(e)

            # Inform users after each trial
            print("Model trial success")

            # Record best performance and seed within these 10 trials
            if cost < lowest_cost:
                lowest_cost = cost
                best_accuracy = accuracy
                best_seed = seed

        # Inform users after each a set of hyperparameters is done testing
        print("10 seeds trialed - entry recorded")

        # Record entry in persistent CSV (./_IO/output/cost_by_hyperparameter.csv)
        new_row = (hp.MOMENTUM, hp.VARIANCE, hp.LEARNING_RATE,
                   best_seed, lowest_cost, best_accuracy)
        with open('_IO/output/cost_by_hyperparameter.csv', 'a', newline = '') as file:
            writer = csv.writer(file)
            writer.writerow(new_row)

if __name__ == "__main__":
    main()