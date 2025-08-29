import numpy as np

# Hyperparameter (beta) for EWMA - should be close to 1
HYPERPARAM : float = 0.94

class EWMA:
    _instances = []
    def __init__(self):
        EWMA._instances.append(self)
        self.weights_average : np.ndarray = np.empty(1)
        self.bias_average : np.ndarray = np.empty(1)

    def add_gradient(self, dW : np.ndarray, db : np.ndarray):
        if not self.weights_average.any():
            self.weights_average = dW
            self.bias_average = db

        else:
            self.weights_average = (self.weights_average * HYPERPARAM) + dW * (1 - HYPERPARAM)
            self.bias_average = (self.bias_average * HYPERPARAM) + db * (1 - HYPERPARAM)

    def get_gradient(self):
        return self.weights_average, self.bias_average

"""
import numpy as np

# Hyperparameter (beta) for EWMA - should be close to 1
MOMENTUM : float = 0.9
VARIANCE : float = 0.999

# Small constant to avoid divide by 0 errors
EPSILON = 10 ** (-15)

class EWMA:
    _instances = []
    def __init__(self):
        EWMA._instances.append(self)
        self.momentum_weights : np.ndarray = np.empty(1)
        self.momentum_bias : np.ndarray = np.empty(1)

        self.variance_weights : np.ndarray = np.empty(1)
        self.variance_bias : np.ndarray = np.empty(1)


    def add_gradient(self, dW : np.ndarray, db : np.ndarray):
        if not self.momentum_weights.any():
            self.momentum_weights = dW
            self.momentum_bias = db

        else:
            self.momentum_weights = (self.momentum_weights * MOMENTUM) + dW * (1 - MOMENTUM)
            self.momentum_bias = (self.momentum_bias * MOMENTUM) + db * (1 - MOMENTUM)

            self.variance_weights = (VARIANCE * self.variance_weights) + dW ** 2 * (1 - VARIANCE)
            self.variance_bias = (VARIANCE * self.variance_bias) + db ** 2 * (1 - VARIANCE)

    def calculate_weights(self):
        final_weights = self.momentum_weights / (self.variance_weights ** 0.5 + EPSILON)
        final_bias = self.momentum_bias / (self.variance_bias ** 0.5 + EPSILON)

        return final_weights, final_bias

    def get_gradient(self):
        final_weights, final_bias = self.calculate_weights()
        return final_weights, final_bias

    @classmethod
    def clear_instances(cls):
        cls._instances.clear()


"""