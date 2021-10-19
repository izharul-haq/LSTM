import numpy as np
import pandas as pd

from model import LSTM


# Read training data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# Preprocess data
# TODO : preprocessing data
# TODO : 1. Drop date feature
# TODO : 2. Handle volume feature
# TODO : 3. Handle market cap feature

# Initialize model
lstm = LSTM(n_input=4, n_hidden=4, n_output=4, timestep=1)
print(lstm.forward(np.array([[0, 1, 2, 3]])))
