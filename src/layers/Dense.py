import numpy as np

from .Layer import Layer
from functions import ACTIVATION


class Dense(Layer):
    '''Simple dense layer implementation'''

    def __init__(self, n_input: int, n_output: int, activation: str):
        # Parameters
        self.__n_input = n_input
        self.__n_output = n_output
        self.__activation = activation

        # Weight Matrix
        self.__W = np.random.randn(n_output, self.__n_input)

        # Biases
        self.__b = np.random.randn(n_output)

        # ACTIVATION FUNCTIONS
        self.__func, self.__func_prime = ACTIVATION[activation]

    def get_output_shape(self) -> tuple:
        '''Get output shape from dense layer'''

        return (None, self.__n_output)

    def get_params(self) -> int:
        '''Get number of parameters from dense layer'''

        return (self.__n_input + 1) * self.__n_output

    def forward(self, _input: np.ndarray) -> np.ndarray:
        '''Execute forward propagation for LSTM layer with
        given input'''

        summ = self.__W @ _input + self.__b

        return self.__func(summ)

    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        '''Execute backward propagation for dense layer
        with given gradient and learning rate'''

        pass

    # TODO : convert dense layer into dictionary
    def to_dict(self) -> dict:
        '''Convert dense layer into dictionary'''
        
        return {
            'name': 'Dense',
            'n_input' : self.__n_input,
            'n_output' : self.__n_output,
            'activation': self.__activation,
            'weights': self.__W.tolist(),
            'biases': self.__b.tolist(),
        }


    # TODO : load dense layer from dictionary
    def from_dict(self, data: dict) -> None:
        '''Load dense layer from dictionary'''
        self.__n_input = data['n_input']
        self.__n_output = data['n_output']
        self.__activation = data['activation']
        self.__func, self.__func_prime = ACTIVATION[data['activation']]
        self.__W = np.array(data['weights'])
        self.__b = np.array(data['biases'])

        Layer._output_shape = self.get_output_shape()

