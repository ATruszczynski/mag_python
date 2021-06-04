from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
import numpy as np


def test_run_relus():
    network = FeedForwardNeuralNetwork(neuronCounts=[2, 2, 2], actFun=[ReLu(), ReLu()], lossFun=CrossEntropy(),
                                       learningRate=0, momCoeff=0, batchSize=0, seed=1001)

    network.weights=[None, np.array([[1, 1], [1, 1]]), np.array([[1, 1], [1, 1]])]
    network.biases=[None, np.array([[1], [1]]), np.array([[1], [1]])]

    result = network.run(np.array([[2], [1]]))

    assert np.array_equal(result, np.array([[9], [9]]))

