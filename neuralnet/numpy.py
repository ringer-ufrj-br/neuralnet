import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inverse_sigmoid(y, epsilon=1e-15):
    # Clips values to avoid division by zero or log of zero
    y = np.clip(y, epsilon, 1 - epsilon)
    return np.log(y / (1 - y))


def alternative_norm1(data):
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data / norms[:, None]


type Numpy1DIntegerArray = np.ndarray[tuple[int], np.integer]
type Numpy1DFloatArray = np.ndarray[tuple[int], np.floating]
