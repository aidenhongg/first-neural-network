import numpy as np

import hyperparams as hp
import neural_network.label as lb

def get_CCE(output_neurons : np.array) -> float:
    """
    Ideal values hot-coded as follows (top - down): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Loss here uses CCE, which is equivalent to -1 * sum(ideal * ln(real))
    An epsilon of 10 **(-15) is used to avoid ln(0) errors.
    """
    # Add epsilon
    output_neurons += hp.EPSILON

    # get the CCE vector
    CCE = lb.current_label * np.log(output_neurons)

    loss = -1 * np.sum(CCE)
    return loss

def get_cost(total_loss : float, training_set_size : int) -> float:
    return total_loss / training_set_size

def ReLU(weighted_sum : np.ndarray) -> np.ndarray:
    activated_sum = np.copy(weighted_sum)
    activated_sum[activated_sum < 0] = 0
    return activated_sum

def softmax(weighted_sum : np.ndarray) -> np.ndarray:
    # Stabilize the sum by subtracting the max value in the array - prevents overflow
    stabilized_sum = weighted_sum - np.max(weighted_sum)
    stabilized_exp = np.exp(stabilized_sum)
    activated_sum = stabilized_exp / np.sum(stabilized_exp)
    return activated_sum
