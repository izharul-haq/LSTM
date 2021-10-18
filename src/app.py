import pandas as pd

from model import LSTM


# Read training data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
