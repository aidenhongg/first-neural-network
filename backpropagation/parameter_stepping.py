import numpy as np

import hyperparams as hp

# Small constant to avoid divide by 0 errors

class EWMA:
    def __init__(self):
        self.momentum_weights : np.ndarray = np.empty(1)
        self.momentum_bias : np.ndarray = np.empty(1)

        self.variance_weights : np.ndarray = np.empty(1)
        self.variance_bias : np.ndarray = np.empty(1)


    def add_gradient(self, dW : np.ndarray, db : np.ndarray):
        if not self.momentum_weights.any():
            self.momentum_weights = dW * (1 - hp.MOMENTUM)
            self.momentum_bias = db * (1 - hp.MOMENTUM)

            self.variance_weights = dW ** 2 * (1 - hp.VARIANCE)
            self.variance_bias = db ** 2 * (1 - hp.VARIANCE)


        else:
            self.momentum_weights = (self.momentum_weights * hp.MOMENTUM) + dW * (1 - hp.MOMENTUM)
            self.momentum_bias = (self.momentum_bias * hp.MOMENTUM) + db * (1 - hp.MOMENTUM)

            self.variance_weights = (hp.VARIANCE * self.variance_weights) + dW ** 2 * (1 - hp.VARIANCE)
            self.variance_bias = (hp.VARIANCE * self.variance_bias) + db ** 2 * (1 - hp.VARIANCE)


    def get_gradient(self):
        momentum_weights_hat = self.momentum_weights / (1 - hp.MOMENTUM ** 2)
        momentum_bias_hat = self.momentum_bias / (1 - hp.MOMENTUM ** 2)

        variance_weights_hat = self.variance_weights / (1 - hp.VARIANCE ** 2)
        variance_weights_hat = np.maximum(variance_weights_hat, 0)
        variance_bias_hat = self.variance_bias / (1 - hp.VARIANCE ** 2)
        variance_bias_hat = np.maximum(variance_bias_hat, 0)

        final_weights = momentum_weights_hat / (variance_weights_hat ** 0.5 + hp.EPSILON)
        final_bias = momentum_bias_hat / (variance_bias_hat ** 0.5 + hp.EPSILON)

        return final_weights, final_bias
