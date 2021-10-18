import numpy as np

from functions import ACTIVATION


class LSTM:
    '''Simple LSTM implementation'''

    def __init__(self, n_cells: int, input_size: (int, int)):
        self.__n_cells = n_cells
        self.__timestep = input_size[0]
        self.__features = input_size[1]

        pass

    # TODO : implement forward propagation
    def forward(self, _input: np.ndarray) -> np.ndarray:
        '''Execute forward propagation given input.

        It's assumed that given input is a list of encoded
        element using one hot encoding'''

        pass

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, learning_rate: float, epochs: int) -> None:
        '''Train LSTM model using backpropagation
        through time (BPTT) using given training
        data and learning rate for given epochs'''

        pass

    # TODO : implement summary method
    def summary(self) -> None:
        '''Print a summary for this model'''

        pass
