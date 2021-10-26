import json

import numpy as np

from layers import Layer


class Sequential:
    def __init__(self, name: str = 'Seq_model', layers: list = None):
        # Identifier
        self.__name = name

        # Attributes
        self.__layers: list[Layer] = layers

    def forward(self, _input: np.ndarray) -> np.ndarray:
        '''Execute forward propagation for Sequential model'''

        res = _input
        for layer in self.layers:
            layer.forward(res)

        return res

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, learning_rate: float, epochs: int) -> None:
        '''Train Sequential model with given training data
        and learning for given epochs'''

        pass

    # TODO : implement summary method
    def summary(self) -> None:
        '''Print a summary for this model'''

        print(f'Model {self.__name}')
        print('------------------------------------------------------------------')
        print()  # TODO : add headers
        print('==================================================================')

        # TODO : print important information (parameter, output shape, etc.)

    # TODO : implemenet save model to an external .json file
    def save(self, filename: str) -> None:
        '''Save sequential model to an external .json file'''

        pass

    # TODO : implemenet load model from an external .json file
    def load(self, filename: str) -> None:
        '''Load sequential model from an external .json file'''

        pass
