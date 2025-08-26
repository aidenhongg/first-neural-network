import numpy as np

def get_loss(label : int, output_neurons : np.array) -> float:
    """
    Ideal values hot-coded as follows (top - down): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    # Instantiate one hot-coded vector
    hot_encoded = np.array([int(i == label) for i in range(10)]).reshape(-1, 1)

    # Ideal (hot-encoded) minus actual vector
    error = hot_encoded - output_neurons

    # Square the error and sum it to get the loss
    squared_error = error ** 2
    loss = np.sum(squared_error)
    return loss

def get_cost(total_loss : float, training_set_size : int) -> float:
    return total_loss / training_set_size

