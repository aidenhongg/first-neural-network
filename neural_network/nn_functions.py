import numpy as np

import hyperparameters_flags as hp
import neural_network.label as lb

"""
nn_functions defines all mathematical functions used in layer calculations.
"""

def get_CCE(output_neurons : np.array) -> float:
    """
    Categorical cross-entropy is the loss function. 
        - Ignores scalars on non-target indices 
        (i.e, if label is '2' ignore all indices but 2)
        - Punishes heavily for deviations from ideal value at target index """""

    # Add small epsilon from hp.EPSILON to avoid divide by 0
    output_neurons += hp.EPSILON

    # Get the CCE vector - hadamard product of ideal output and log of real output
    CCE = lb.current_label * np.log(output_neurons)

    # Sum the total loss and negate to aid in gradient descent
    loss = -1 * np.sum(CCE)
    return loss

def get_cost(total_loss : float, training_set_size : int) -> float:
    """
    Return the cost (average loss) based on total loss and dataset size """""
    return total_loss / training_set_size

def ReLU(weighted_sum : np.ndarray) -> np.ndarray:
    """
    Return the ReLU activated vector - zero out all negative values """""

    # Make a copy to avoid altering the original weighted sum
    activated_sum = np.copy(weighted_sum)
    activated_sum[activated_sum < 0] = 0
    return activated_sum

def softmax(weighted_sum : np.ndarray) -> np.ndarray:
    """
    Return the Softmaxxed vector 
        - Let b be the sum of e to the power of each value in the vector
        - Then return the given weighted sum vector / b """""

    # Stabilize the sum by subtracting the max value in the array - prevents overflow
    stabilized_sum = weighted_sum - np.max(weighted_sum)

    # Apply Softmax
    stabilized_exp = np.exp(stabilized_sum)
    activated_sum = stabilized_exp / np.sum(stabilized_exp)
    return activated_sum
