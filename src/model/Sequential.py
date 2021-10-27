import json

import numpy as np


from errors import InputError
from layers import LSTM, Dense
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

        model = {
            'name': self.__name,
            'layers': [layer.to_dict() for layer in self.__layers],
        }

        json_obj = json.dumps(model, indent=4)
        with open(f'{filename}.json', 'w') as f:
            f.write(json_obj)

    # TODO : implemenet load model from an external .json file
    def load(self, filename: str) -> None:
        '''Load sequential model from an external .json file'''

        with open(f'{filename}.json', 'r') as f:
            json_obj = json.load(f)

            self.__name = json_obj['name']
            self.__layers = []
            for layer in json_obj['layers']:
                if layer['name'] == 'LSTM':
                    temp = LSTM(n_input=4, n_hidden=4, n_output=4, timestep=1)     # Dummy layer

                elif layer['name'] == 'Dense':
                    temp = Dense(n_input = 1, n_output =1 , activation = "sigmoid")          # Dummy layer

                else:
                    raise InputError('layer not supported')

                temp.from_dict(layer)
                self.__layers.append(temp)

            f.close()
