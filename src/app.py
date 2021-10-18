import pandas as pd

from model import LSTM


# Read training data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# Preprocess data
# TODO : preprocessing data


# Initialize model
lstm = LSTM(50, input_size=(32, 6))
