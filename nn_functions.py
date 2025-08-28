import numpy as np
import neural_network as nn

def get_CCE(output_neurons : np.array) -> float:
    """
    Ideal values hot-coded as follows (top - down): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Loss here uses CCE, which is equivalent to -1 * sum(ideal * ln(real))
    An epsilon of 10 **(-15) is used to avoid ln(0) errors.
    """
    # Add epsilon
    epsilon = 10 **(-15)
    output_neurons += epsilon

    # get the CCE vector
    CCE = nn.current_label * np.log(output_neurons)

    loss = -1 * np.sum(CCE)
    return loss

def get_cost(total_loss : float, training_set_size : int) -> float:
    return total_loss / training_set_size

