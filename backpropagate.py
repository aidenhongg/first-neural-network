import neural_network as nn
import numpy as np

"""
derived functions to be refactored
"""

def backpropagation(L  : nn.Layer):
    if L == nn.LAYERS[-1]:
        dC_dw = CCE_softmax_der(L) @ dz_dw(L)





def RelU_der(weighted_sum : np.ndarray) -> np.ndarray:
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

print(nn.current_label)
nn.update_label(4)
print(nn.current_label)