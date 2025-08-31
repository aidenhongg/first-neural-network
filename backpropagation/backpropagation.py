import numpy as np

import neural_network as nn
from neural_network import label as lb


error_signal : np.ndarray = None
next_layer : nn.Layer = None

def backpropagate(L : nn.Layer) -> tuple[np.ndarray, np.ndarray]:
    global error_signal
    global next_layer
    if L == nn.LAYERS[-1]:
        error_signal = CCE_softmax_der(L)
        dC_dw = error_signal @ dz_dw(L.previous_layer)

    else:
        weighted_sum = L.get_weighted_sum()
        error_signal = (next_layer.weights.T @ error_signal) * ReLU_der(weighted_sum)
        dC_dw = error_signal @ dz_dw(L.previous_layer)

    # This layer is now the successive layer in the NN to the
    # next layer that will be backpropagated
    next_layer = L
    # Error signal is equivalent to db
    return dC_dw, error_signal

def ReLU_der(weighted_sum : np.ndarray) -> np.ndarray:
    derived_sum = np.copy(weighted_sum)
    derived_sum[derived_sum <= 0] = 0
    derived_sum[derived_sum > 0] = 1

    return derived_sum

def CCE_softmax_der(L : nn.Layer) -> np.ndarray:
    # Using Softmax & CCE on the last layer greatly simplifies the derivative
    return L.neurons - lb.current_label

def dz_dw(L : nn.Layer) -> np.ndarray:
     return L.neurons.T
