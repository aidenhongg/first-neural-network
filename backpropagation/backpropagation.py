import numpy as np

import neural_network as nn
from neural_network import label as lb

"""
backpropagation.py handles backpropagation on a single layer and its parameters.
    - ReLU is used on all hidden layers except the output
    - The output layer uses Softmax + CCE to calculate loss """""

# error_signal tracks the previously calculated error signal at moment T - 1.
error_signal : np.ndarray = None

# next_layer tracks the layer from which the error signal at moment T - 1 was calculated.
next_layer : nn.Layer = None

def backpropagate(L : nn.Layer) -> tuple[np.ndarray, np.ndarray]:
    """
    Backpropagates through a single layer after a forward feed. 
        - Moment T, T - 1 refers to the current and previous backpropagation iteration
        - Error signals calculated recursively based on error signal at T - 1
        - next_layer stores the layer that was backpropagated at T - 1 
        - Gradient is calculated based on the error signal at moment T """""
    global error_signal
    global next_layer

    # Instantiate the error signal if T = 1 (first backpropagation on the last layer)
    if L == nn.LAYERS[-1]:
        # Find error signal based on CCE and softmax derivative
        error_signal = CCE_softmax_der(L)

        # Find gradient from error signal and weighted sum derivative
        dC_dw = error_signal @ dz_dw(L)

    # On T > 1 find error signal recursively (backpropagating any layer but the last)
    else:
        # Find error signal based on weights and error signal from moment T - 1
        # (previously backpropagated layer)
        # Then multiply against ReLU derivative of current layer's weighted sum
        weighted_sum = L.get_weighted_sum()
        error_signal = (next_layer.weights.T @ error_signal) * ReLU_der(weighted_sum)

        # Find gradient from error signal and weighted sum derivative
        dC_dw = error_signal @ dz_dw(L)

    # Current layer becomes moment T - 1 layer for the next iteration
    next_layer = L

    # Return cost gradient for weights and bias (error signal is equivalent to dC/db)
    return dC_dw, error_signal

def ReLU_der(weighted_sum : np.ndarray) -> np.ndarray:
    """
    Apply the ReLU derivative to an entire array, and return the result.
        -ReLU's derivative is 1 for all positive values 0 for all negatives
        -Technically undefined at 0 but we'll just make it 0 at 0 """""

    # Copy weighted sum to not affect original
    derived_sum = np.copy(weighted_sum)

    # Filter all negative and 0 values and assign 0 value
    derived_sum[derived_sum <= 0] = 0
    # Filter all positive values and assign 1
    derived_sum[derived_sum > 0] = 1

    return derived_sum

def CCE_softmax_der(L : nn.Layer) -> np.ndarray:
    """
    Apply the (CCE)' * (Softmax)' for backpropagating the last layer
        - Multiplying these derivatives simplifies them greatly. """""
    # Simplifies to the difference between the real and ideal output
    return L.neurons - lb.current_label

def dz_dw(L : nn.Layer) -> np.ndarray:
    """
    Apply the derivative of the weighted sum function
        - Because it is linear - multiplied against the neurons of the previous layer -
         the neurons themselves are returned.
        - The neurons are transposed to copy them, avoiding side effects, and to 
        aid multiplication. """""
    return L.previous_layer.neurons.T
