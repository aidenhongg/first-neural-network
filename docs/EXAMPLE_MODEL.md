Using the hyperparameters mentioned in `README.md`, I was able to achieve a cost of 0.23 and an accuracy of 94.5% on some seeds.

This is a unicorn model I instantiated while messing around with my model, which gave me a 0.22 cost and 95.11% accuracy.

```from mnist import MNIST

# Dataset for modules to call universally
RAW_DATA : MNIST = MNIST()

"""
hyperparameters_flags.py holds hyperparameters and flags for users to adjust
before training, creating, and predicting data using their model. """

# Batch size - # of examples analyzed between each gradient adjustment
BATCH_SIZE : int = 32

# Learning rate of the model
LEARNING_RATE : float = 0.0009

# Patience - stop training if cost does not decrease by PATIENCE_BUFFER amount
# for PATIENCE # of epochs
PATIENCE : int = 20
PATIENCE_BUFFER : float = 0.00001

# Load past seed conditions for testing - fixing random initialization of parameters
LOAD_SEED : bool = True
# Set custom seed to be used if LOAD_SEED = True
SEED : int = 968141384

# Load previously stored parameters - weights and biases - if True
LOAD_MODEL : bool = False

# Save trained model parameters - weights and biases - if True
SAVE_MODEL : bool = True

# Momentum and variation smoothing rates for ADAM optimization
# MOMENTUM is beta1
# VARIANCE is beta2
MOMENTUM : float = 0.95
VARIANCE : float = 0.997

# EPSILON offset adjustments when preventing divide by 0 errors
# A smaller EPSILON causes smaller deviations from true outputs but takes longer to calculate
EPSILON : float = 10**(-16)
```
