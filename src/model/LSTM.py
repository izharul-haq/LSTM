import numpy as np

from functions import ACTIVATION


class LSTM:
    '''Simple LSTM implementation'''

    def __init__(self, n_cells: int, input_size: (int, int)):
        self.__n_cells = n_cells
        self.__input_size = input_size

        pass

    # TODO : implement forward propagation
    def forward(self, _input: np.ndarray) -> np.ndarray:
        '''Execute forward propagation given input.

        It's assumed that given input is a list of encoded
        element using one hot encoding'''

        pass
