from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):

    @abstractmethod
    def forward(self, _input: np.ndarray) -> np.ndarray:
        '''Execute forward propagation for this layer'''

        pass

    @abstractmethod
    def get_output_shape(self) -> tuple:
        '''Get output shape for this layer'''

        pass

    @abstractmethod
    def get_params(self) -> int:
        '''Get number of parameters for this layer'''

        pass

    @abstractmethod
    def backward(self, grad: np.ndarray, learning_rate: float) -> np.ndarray:
        '''Execute backward propagation for this layer'''

        pass

    @abstractmethod
    def to_dict(self) -> dict:
        '''Convert this layer into a dictionary'''

        pass

    @abstractmethod
    def from_dict(self, data: dict) -> None:
        '''Load this layer from a dictionary'''

        pass
