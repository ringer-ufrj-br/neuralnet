import numpy as np


def inverse_sigmoid(y):
    # Clips values to avoid division by zero or log of zero
    y = np.clip(y, 1e-15, 1 - 1e-15)
    return np.log(y / (1 - y))
