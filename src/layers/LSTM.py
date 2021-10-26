import numpy as np

from .Layer import Layer
from functions import sigmoid, tanh


class LSTM(Layer):
    '''Simple Many-to-One LSTM Layer implementation'''

    idx = 0

    def __init__(self,
                 n_input: int,
                 n_hidden: int,
                 n_output: int,
                 timestep: int
                 ):
        self.__name = f'LSTM {LSTM.idx + 1}'
        LSTM.idx += 1

        # Parameters
        self.__n_input = n_input        # number of input neurons; number of features
        self.__n_hidden = n_hidden      # number of hidden neurons
        self.__n_output = n_output      # number of output neurons

        self.__timestep = timestep      # timestep used

        # Weight Matrices
        self.__Uf = np.random.randn(n_hidden, self.__n_input)
        self.__Ui = np.random.randn(n_hidden, self.__n_input)
        self.__Uc = np.random.randn(n_hidden, self.__n_input)
        self.__Uo = np.random.randn(n_hidden, self.__n_input)

        self.__Wf = np.random.randn(n_hidden, n_hidden)
        self.__Wi = np.random.randn(n_hidden, n_hidden)
        self.__Wc = np.random.randn(n_hidden, n_hidden)
        self.__Wo = np.random.randn(n_hidden, n_hidden)

        self.__V = np.random.randn(n_output, n_hidden)

        # Biases
        self.__bf = np.random.randn(n_hidden)
        self.__bi = np.random.randn(n_hidden)
        self.__bc = np.random.randn(n_hidden)
        self.__bo = np.random.randn(n_hidden)

        # biases from RNN state to output vector
        self.__biases_ho = np.random.randn(n_output)

    # TODO : implement get output shapes
    def get_output_shape(self):
        '''Get output shape from LSTM layer'''

        pass

    # TODO : implement get parameters
    def get_params(self):
        '''Get number of parameters from dense layer'''

        pass

    # TODO : review forward propagation
    def forward(self, _input: np.ndarray) -> np.ndarray:
        '''Execute forward propagation for LSTM layer with
        given input'''

        prev_h = np.zeros(self.__n_hidden)
        prev_c = np.zeros(self.__n_hidden)

        for i in range(self.__timestep):
            # Forget Gate
            curr_f = sigmoid(
                self.__Uf @ _input[i] + self.__Wf @ prev_h + self.__bf
            )

            # Input Gate
            curr_i = sigmoid(
                self.__Ui @ _input[i] + self.__Wi @ prev_h + self.__bi
            )

            temp_c = tanh(
                self.__Uc @ _input[i] + self.__Wc @ prev_h + self.__bc
            )

            # Cell State
            prev_c = np.multiply(curr_f, prev_c) + np.multiply(curr_i, temp_c)

            # Output Gate
            curr_o = sigmoid(
                self.__Uo @ _input[i] + self.__Wo @ prev_h + self.__bo
            )

            prev_h = np.multiply(curr_o, tanh(prev_c))

        # output neurons use linear activation
        return prev_h

    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        '''Execute backward propagation for LSTM layer
        with given gradient and learning rate'''

        pass

    # TODO : convert LSTM layer into dictionary
    def to_dict(self) -> dict:
        '''Convert LSTM layer into dictionary'''

        pass

    # TODO : load LSTM layer from dictionary
    def from_dict(self, data: dict) -> None:
        '''Load LSTM layer from dictionary'''

        pass
