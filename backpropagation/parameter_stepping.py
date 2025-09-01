import numpy as np

import hyperparameters_flags as hp

"""
parameter_stepping holds the exponential weighted moving averages (EWMA) 
of each parameter. These persist throughout the training of a given model.
    - EWMAs are updated with the averages of all gradients in a batch
    - EWMAs impact the weight of each batch on training, based on their recency, and
     the magnitude of gradients. 

We are using ADAM to calculate the final gradients."""

class EWMA:
    """
    One unique EWMA object is instantiated for each layer, storing updates for its 
    attached weights and biases.
        - Each EWMA technically holds 4 exponential weighted moving averages:
        1. One each for the momentum of the weights and biases 
        2. One each for the variance of the weights and biases. 
        
    beta1 is momentum smoothing factor, given by hp.MOMENTUM
    beta 2 is variance smoothing factor, given by hp.VARIANCE """""

    def __init__(self):
        # Instantiate all four exponential weighted moving averages to None
        self.momentum_weights : np.ndarray = None
        self.momentum_bias : np.ndarray = None

        self.variance_weights : np.ndarray = None
        self.variance_bias : np.ndarray = None

        self.time = 0

    def add_gradient(self, dW : np.ndarray, db : np.ndarray):
        # If no gradients have ever been passed before, instantiate the base case
        if self.momentum_weights is None:
            # Momentum is the gradient * (1 - beta1)
            self.momentum_weights = dW * (1 - hp.MOMENTUM)
            self.momentum_bias = db * (1 - hp.MOMENTUM)

            # Variance is calculated as the gradient squared times (1 - beta2)
            self.variance_weights = dW ** 2 * (1 - hp.VARIANCE)
            self.variance_bias = db ** 2 * (1 - hp.VARIANCE)

        # Otherwise, apply the recursive case
        else:
            # Momentum is the sum of:
            # previous momentum * beta1, and the given gradient * (1 - beta1)
            self.momentum_weights = (self.momentum_weights * hp.MOMENTUM) + dW * (1 - hp.MOMENTUM)
            self.momentum_bias = (self.momentum_bias * hp.MOMENTUM) + db * (1 - hp.MOMENTUM)

            # Variance is the sum of:
            # previous momentum * beta2, and the given gradient squared * (1 - beta2)
            self.variance_weights = (hp.VARIANCE * self.variance_weights) + dW ** 2 * (1 - hp.VARIANCE)
            self.variance_bias = (hp.VARIANCE * self.variance_bias) + db ** 2 * (1 - hp.VARIANCE)

        self.time += 1


    def get_gradient(self):
        # Return the final gradient that will update the parameters.

        # Apply bias correction to get new momentum
        # given by original momentum / 1 - (beta1 ^ time). As time increases:
        #   1. (beta1 ^ time) approaches 0
        #   2. 1 - (beta1 ^ time) approaches 1,
        #   3. New momentum approaches old momentum
        # Thus, this avoids initial gradients being weighted too heavily.
        momentum_weights_hat = self.momentum_weights / (1 - hp.MOMENTUM ** self.time)
        momentum_bias_hat = self.momentum_bias / (1 - hp.MOMENTUM ** self.time)

        # Apply equivalent operation to get new variances
        variance_weights_hat = self.variance_weights / (1 - hp.VARIANCE ** self.time)

        # Mathematically variance is squared and always positive, but computational errors
        # can cause negative values.
        # Thus, zero out negative values before final calculation.
        variance_weights_hat = np.maximum(variance_weights_hat, 0)
        variance_bias_hat = self.variance_bias / (1 - hp.VARIANCE ** self.time)
        variance_bias_hat = np.maximum(variance_bias_hat, 0)

        # Final weights and bias gradients are given by:
        # momentum / sqrt(variance)
        # Small EPSILON from hp.EPSILON added to denominator to avoid divide by 0 errors
        final_weights = momentum_weights_hat / (variance_weights_hat ** 0.5 + hp.EPSILON)
        final_bias = momentum_bias_hat / (variance_bias_hat ** 0.5 + hp.EPSILON)

        return final_weights, final_bias
