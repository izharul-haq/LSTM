import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    '''Calculate tanh activation value from
    given numpy array'''

    return np.tanh(x)


def tanh_prime(x: np.ndarray) -> np.ndarray:
    '''Calculate differential of tanh activation
    value from given numoy array'''

    return 1 - (np.tanh(x) ** 2)
