import numpy as np

def get_CCE(label : int, output_neurons : np.array) -> float:
    """
    Ideal values hot-coded as follows (top - down): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Loss here uses CCE, which is equivalent to -1 * sum(ideal * ln(real))
    An epsilon of 10 **(-15) is used to avoid ln(0) errors.
    """
    # Add epsilon
    epsilon = 10 **(-15)
    output_neurons += epsilon

    # Instantiate one hot-coded vector
    hot_encoded = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)

    # get the CCE vector
    CCE = hot_encoded * np.log(output_neurons)

    loss = -1 * np.sum(CCE)
    return loss

def get_cost(total_loss : float, training_set_size : int) -> float:
    return total_loss / training_set_size

