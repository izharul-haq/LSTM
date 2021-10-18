import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    '''Calculate sigmoid activation value from
    given numpy array'''

    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.ndarray) -> np.ndarray:
    '''Calculate differential of sigmoid activation
    value from given numpy array'''

    s = sigmoid(x)
    return s * (1 - s)
