import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from layers import LSTM, Dense
from model import Sequential

# Read training data
train_ancestor = pd.read_csv('../data/train.csv')
test_ancestor = pd.read_csv('../data/test.csv')


# Preprocess data
# TODO : preprocessing data
# TODO : 1. Drop date feature
train = train_ancestor.drop(['Date'], axis=1)
test = test_ancestor.drop(['Date'], axis=1)

# Only 32 date time before
train.drop(train.index[32:],0,inplace=True)

# Balik urutan Row
train = train.loc[::-1]

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
# train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
# test_scaled = pd.DataFrame(test_scaled, columns=test.columns)

# Initialize model
model = Sequential(layers=[
    LSTM(n_input=6, n_hidden=6, timestep=32),
    Dense(n_input = 6, n_output =6 , activation = "sigmoid"),
])
# model.save('json/example3')
# lstm2 = LSTM()
# model.load('json/example3')
# print(train_scaled.shape)
print(model.forward(train_scaled))
# print(train_scaled)
model.summary()