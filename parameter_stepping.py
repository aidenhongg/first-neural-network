
import numpy as np

# Hyperparameter (beta) for EWMA - should be close to 1
MOMENTUM : float = 0.94
VARIANCE : float = 0.99

# Small constant to avoid divide by 0 errors
EPSILON = 10 ** (-100)

class EWMA:
    def __init__(self):
        self.momentum_weights : np.ndarray = np.empty(1)
        self.momentum_bias : np.ndarray = np.empty(1)

        self.variance_weights : np.ndarray = np.empty(1)
        self.variance_bias : np.ndarray = np.empty(1)


    def add_gradient(self, dW : np.ndarray, db : np.ndarray):
        if not self.momentum_weights.any():
            self.momentum_weights = dW * (1 - MOMENTUM)
            self.momentum_bias = db * (1 - MOMENTUM)

            self.variance_weights = dW ** 2 * (1 - VARIANCE)
            self.variance_bias = db ** 2 * (1 - VARIANCE)


        else:
            self.momentum_weights = (self.momentum_weights * MOMENTUM) + dW * (1 - MOMENTUM)
            self.momentum_bias = (self.momentum_bias * MOMENTUM) + db * (1 - MOMENTUM)

            self.variance_weights = (VARIANCE * self.variance_weights) + dW ** 2 * (1 - VARIANCE)
            self.variance_bias = (VARIANCE * self.variance_bias) + db ** 2 * (1 - VARIANCE)


    def get_gradient(self):
        momentum_weights_hat = self.momentum_weights / (1 - MOMENTUM ** 2)
        momentum_bias_hat = self.momentum_bias / (1 - MOMENTUM ** 2)

        variance_weights_hat = self.variance_weights / (1 - VARIANCE ** 2)
        variance_bias_hat = self.variance_bias / (1 - VARIANCE ** 2)

        final_weights = momentum_weights_hat / ((variance_weights_hat + EPSILON) ** 0.5 + EPSILON)
        final_bias = momentum_bias_hat / ((variance_bias_hat + EPSILON) ** 0.5 + EPSILON)

        return final_weights, final_bias
