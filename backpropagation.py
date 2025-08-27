import neural_network as nn
import numpy as np

"""
derived functions to be refactored
"""
error_signal = 0
next_layer = None

def backpropagate(L  : nn.Layer):
    global error_signal
    global next_layer

    if L == nn.LAYERS[-1]:
        error_signal = CCE_softmax_der(L)
        dC_dw = error_signal @ dz_dw(L)

    else:
        weighted_sum = L.get_weighted_sum()
        error_signal = (next_layer.weights.T @ error_signal) * ReLU_der(weighted_sum)
        dC_dw = error_signal @ dz_dw(L)

    return dC_dw



def ReLU_der(weighted_sum : np.ndarray) -> np.ndarray:
    derived_sum = np.copy(weighted_sum)
    derived_sum[derived_sum <= 0] = 0
    derived_sum[derived_sum > 0] = 1

    return derived_sum

def CCE_softmax_der(L : nn.Layer) -> np.ndarray:
    # Using Softmax & CCE on the last layer greatly simplifies the derivative
    return L.neurons - nn.current_label

def dz_dw(L : nn.Layer) -> np.ndarray:
     L_sub_1 = L.previous_layer
     L_sub_T = L_sub_1.neurons.T
     return L_sub_T