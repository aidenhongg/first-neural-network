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

# Load past-initialized weights for testing
LOAD_SEED : bool = False
SEED : int = 0
