from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
import numpy as np


def test_run_relus():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=2, hidden_neuron_counts=[2], actFuns=[ReLu(), ReLu()], weights=[np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]])],
                                       biases=[np.array([[1], [1]]), np.array([[1], [1]])], seed=1001)


    result = network.run(np.array([[2], [1]]))

    assert np.array_equal(result, np.array([[9], [9]]))

