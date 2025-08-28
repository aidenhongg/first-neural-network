import numpy as np

# Hyperparameter (beta) for EWMA - should be close to 1
HYPERPARAM : float = 0

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

    @classmethod
    def clear_instances(cls):
        cls._instances.clear()
