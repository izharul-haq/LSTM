import numpy as np


def linear(x: np.ndarray) -> np.ndarray:
    '''Calculate linear activation value from
    given numpy array'''

    return x


def linear_prime(x: np.ndarray) -> np.ndarray:
    '''Calculate differential of linear activation
    value from given numpy array'''

    shape = x.shape

    return np.ones((shape))
