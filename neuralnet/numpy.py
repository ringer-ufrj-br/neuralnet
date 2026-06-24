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


def fixed_point_quantization_alternative_norm1(data, i0, f0):
    scale = 2**f0
    quantized_data = np.round(data * scale).astype(np.int32)
    quantized_data = np.clip(
        quantized_data, -(2 ** (i0 + f0 - 1)), 2 ** (i0 + f0 - 1) - 1
    )
    return quantized_data / scale


type Numpy1DIntegerArray = np.ndarray[tuple[int], np.integer]
type Numpy1DFloatArray = np.ndarray[tuple[int], np.floating]
