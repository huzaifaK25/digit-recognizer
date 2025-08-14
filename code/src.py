import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./data/train.csv')
data = np.array(data)

# m is num of rows and n is num of columns(pixels: 783 total) + 1(label column)
m, n = data.shape
# random shuffle data
np.random.shuffle(data)

# slpit data to dev and train
data_dev = data[0:1000].T # .T = transpose data to make each pizel a row 
Y_dev = data_dev[0] # first row
X_dev = data_dev[1:n] # all columns

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

print(X_train[:, 0].shape)