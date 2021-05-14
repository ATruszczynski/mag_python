import numpy as np
from neural_network import *
from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4])
y = np.array([2, 5, 5, 8])

x = x.reshape(-1, 1)
y = y.reshape(-1, 1)

reg = LinearRegression().fit(x, y)

print(reg.coef_)