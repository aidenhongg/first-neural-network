from mnist import MNIST

# Layer count
LAYER_COUNT = 4

# Batch size between each gradient adjustment
BATCH_SIZE : int = 32

# Step size
STEP_SIZE : float = 0.0001

# Dataset
RAW_DATA : MNIST = MNIST()

# Patience
PATIENCE : int = 5
PATIENCE_BUFFER : float = 0.001

# Load past-initialized weights for testing
LOAD_SEED : bool = False
SEED : int = 1922815607

# Momentum and variation smoothing rates for ADAM optimization
# MOMENTUM should be close to 1
# VARIANCE should be closer to 1 and generally larger
MOMENTUM : float = 0.94
VARIANCE : float = 0.99

EPSILON : float = 10**(-15)

