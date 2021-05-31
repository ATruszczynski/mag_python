import numpy as np
from neural_network import *
from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
from sklearn.linear_model import LinearRegression

arr = np.array([[1, 2], [1.9, 4], [2.1, -2]])
print(np.minimum(arr, 2))