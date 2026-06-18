import numpy as np


def inverse_sigmoid(y):
    # Clips values to avoid division by zero or log of zero
    y = np.clip(y, 1e-15, 1 - 1e-15)
    return np.log(y / (1 - y))


def alternative_norm1(data):
    norms = np.abs(data.sum(axis=1))
    norms[norms == 0] = 1
    return data / norms[:, None]


type Numpy1DIntegerArray = np.ndarray[tuple[int], np.integer]
type Numpy1DFloatArray = np.ndarray[tuple[int], np.floating]
