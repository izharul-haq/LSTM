import numpy as np

from functions import sigmoid, tanh


class LSTM:
    '''Simple Many-to-One LSTM implementation'''

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

    # TODO : review forward propagation
    def forward(self, _input: np.ndarray) -> np.ndarray:
        '''Execute forward propagation given input.

        It's assumed that given input is a list of encoded
        element using one hot encoding'''

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
        return self.__V @ prev_h + self.__biases_ho

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray, learning_rate: float, epochs: int) -> None:
        '''Train LSTM model using backpropagation
        through time (BPTT) using given training
        data and learning rate for given epochs'''

        pass

    # TODO : implement summary method
    def summary(self) -> None:
        '''Print a summary for this model'''

        print(f'Model {self.__name}')
        print('------------------------------------------------------------------')
        print()  # TODO : add headers
        print('==================================================================')

        # TODO : print important information (parameter, output shape, etc.)

    # TODO : implement save model to an external file
    def save(self, filename: str) -> None:
        '''Save LSTM model into an external .json file'''

        pass

    # TODO : implement load model from an external file
    def load(self, filename: str) -> None:
        '''Load LSTM model from an external .json file'''

        pass
