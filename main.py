import numpy as np
from neural_network import *
from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
from sklearn.linear_model import LinearRegression

rl = ReLu()
print(rl.computeDer(np.array([0, 1, -2]).reshape(-1, 1)))

sm = Softmax()
smr = sm.computeDer(np.array([1, 2, 3]).reshape(-1, 1))
print(smr)
print(np.sum(smr, axis=0))
print(np.sum(smr, axis=1))

sg = Sigmoid()
sgr = sg.computeDer(np.array([1, 2, 3]).reshape(-1, 1))

print(sgr)