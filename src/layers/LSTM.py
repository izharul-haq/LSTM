import numpy as np

from .Layer import Layer
from functions import sigmoid, tanh


class LSTM(Layer):
    '''Simple Many-to-One LSTM Layer implementation'''

    def __init__(self,
                 n_input: int,
                 n_hidden: int,
                 timestep: int
                 ):

        # Parameters
        self.__n_input = n_input        # number of input neurons; number of features
        self.__n_hidden = n_hidden      # number of hidden neurons

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

        # Biases
        self.__bf = np.random.randn(n_hidden)
        self.__bi = np.random.randn(n_hidden)
        self.__bc = np.random.randn(n_hidden)
        self.__bo = np.random.randn(n_hidden)

    def get_output_shape(self):
        '''Get output shape from LSTM layer'''

        return (None, self.__n_hidden)

    def get_params(self):
        '''Get number of parameters from dense layer'''

        return 4 * self.__n_hidden * (self.__n_input + self.__n_hidden + 1)

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

    def to_dict(self) -> dict:
        '''Convert LSTM layer into dictionary'''

        return {
            'name': 'LSTM',
            'n_input': self.__n_input,
            'n_hiden': self.__n_hidden,
            'timestep': self.__timestep,
            'Uf': self.__Uf.tolist(),
            'Ui': self.__Ui.tolist(),
            'Uc': self.__Uc.tolist(),
            'Uo': self.__Uo.tolist(),
            'Wf': self.__Wf.tolist(),
            'Wi': self.__Wi.tolist(),
            'Wc': self.__Wc.tolist(),
            'Wo': self.__Wo.tolist(),
            'Bf': self.__bf.tolist(),
            'Bi': self.__bi.tolist(),
            'Bc': self.__bc.tolist(),
            'Bo': self.__bo.tolist(),
        }

        pass

    def from_dict(self, data: dict) -> None:
        '''Load LSTM layer from dictionary'''

        self.__n_input = data['n_input']
        self.__n_hidden = data['n_hiden']

        self.__timestep = data['timestep']

        # Weight Matrices
        self.__Uf = np.array(data['Uf'])
        self.__Ui = np.array(data['Ui'])
        self.__Uc = np.array(data['Uc'])
        self.__Uo = np.array(data['Uo'])

        self.__Wf = np.array(data['Wf'])
        self.__Wi = np.array(data['Wi'])
        self.__Wc = np.array(data['Wc'])
        self.__Wo = np.array(data['Wo'])

        # Biases
        self.__bf = np.array(data['Bf'])
        self.__bi = np.array(data['Bi'])
        self.__bc = np.array(data['Bc'])
        self.__bo = np.array(data['Bo'])
