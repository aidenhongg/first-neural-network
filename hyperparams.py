from mnist import MNIST

# Dataset for modules to call universally
RAW_DATA : MNIST = MNIST()

# Batch size between each gradient adjustment
BATCH_SIZE : int = 32

# Learning rate of the model
LEARNING_RATE : float = 0.0001

# Patience - stop training if cost does not decrease for this many epochs
PATIENCE : int = 5
PATIENCE_BUFFER : float = 0.0001

# Load past conditions for testing
# Seeds initialize the model
LOAD_SEED : bool = True
SEED : int = 714094545

LOAD_MODEL : bool = False
SAVE_MODEL : bool = True

# Momentum and variation smoothing rates for ADAM optimization
# MOMENTUM should be close to 1
# VARIANCE should be closer to 1 and generally larger
MOMENTUM : float = 0.94
VARIANCE : float = 0.99

# EPSILON offset adjustments
EPSILON : float = 10**(-15)

