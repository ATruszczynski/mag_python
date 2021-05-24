import numpy as np
from neural_network import *
from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
from sklearn.linear_model import LinearRegression

from neural_network.LooseNet import LooseNetwork

link_mat = np.zeros((6, 6))
link_mat[0, 2] = 1
link_mat[1, 3] = 1
link_mat[2, 5] = 1
link_mat[2, 3] = 1
link_mat[3, 5] = 1
link_mat[4, 5] = 1

wei = np.zeros((6, 6))
wei[0, 2] = 1
wei[1, 3] = 1
wei[2, 5] = 1
wei[2, 3] = 1
wei[3, 5] = 1
wei[4, 5] = 1

bias = np.zeros((6, 1))
acts = [ReLu(), ReLu(), ReLu(), ReLu(), ReLu(), ReLu()]

random.seed(10011010)


# net = LooseNetwork(input_size=2, output_size=1, links=link_mat, weights=wei, biases=bias, actFuns=acts)
# net.analyse()
# print(net.run(np.array([[1], [1]])))
# print(net.density())
#
# print(net.get_indices_of_no_output_neurons())
# print(net.get_indices_of_no_input_neurons())
# print(net.get_indices_of_disconnected_neurons())
# print(net.to_string())
#
# hrange = HyperparameterRange(neuronCount=5, actFuns=[ReLu(), Sigmoid()])
# pop = generate_population(hrange, 3, 2, 1)
#
# ori = 1

# array = np.array([[-1, 1], [-1, 1]])
# vector = np.vectorize(lambda f, x: f.compute(x), otypes=[object])
# vector([ReLu(), Sigmoid()], array)