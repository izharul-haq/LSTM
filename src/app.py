import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from layers import LSTM, Dense
from model import Sequential

# Read training data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')


# Preprocess data
# TODO : preprocessing data
# TODO : 1. Drop date feature
train = train.drop(['Date'], axis=1)
test = test.drop(['Date'], axis=1)
# TODO : 2. Handle volume feature

# Drop null value and change dtypes of volume to int64
train['Volume'] = train['Volume'].str.replace(',', '')
train['Volume'] = train['Volume'].str.replace('-', '')
test['Volume'] = test['Volume'].str.replace(',', '')
test['Volume'] = test['Volume'].str.replace('-', '')
train = train[train['Volume'] != '']
test = test[test['Volume'] != '']
train['Volume'] = pd.array(train['Volume'], dtype='int64')
test['Volume'] = pd.array(test['Volume'], dtype='int64')

# TODO : 3. Handle market cap feature
# change dtypes of market cap to int64
train['Market Cap'] = train['Market Cap'].str.replace(',', '')
train['Market Cap'] = train['Market Cap'].str.replace('-', '')
test['Market Cap'] = test['Market Cap'].str.replace(',', '')
test['Market Cap'] = test['Market Cap'].str.replace('-', '')
train['Market Cap'] = pd.array(train['Market Cap'], dtype='int64')
test['Market Cap'] = pd.array(test['Market Cap'], dtype='int64')

# Minmax Scaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.fit_transform(test)
train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
test_scaled = pd.DataFrame(test_scaled, columns=test.columns)
# print(train_scaled.head())
# print(test_scaled.head())

# Initialize model
model = Sequential(layers = [
    LSTM(n_input=4, n_hidden=4, timestep=1),
    Dense(n_input = 1, n_output =1 , activation = "sigmoid"),
])
# model.save('json/example3')
# lstm2 = LSTM()
# model.load('json/example3')
# print(lstm.forward(np.array([[0, 1, 2, 3]])))
model.summary()