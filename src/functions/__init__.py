from .sigmoid import sigmoid, sigmoid_prime
from .tanh import tanh, tanh_prime

ACTIVATION = {
    'sigmoid': (sigmoid, sigmoid_prime),
    'tanh': (tanh, tanh_prime),
}
