from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
import numpy as np


def test_run_relus():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=2, hiddenLayerCount=1, neuronCount=1, actFun=ReLu(),
                                       aggrFun=ReLu(), lossFun=QuadDiff(), learningRate=1, momCoeffL=2, seed=1001)

    network.weights[1] = np.array([[1, 1], [1, 1]])
    network.weights[2] = np.array([[1, 1], [1, 1]])
    network.biases[1] = np.array([[1], [1]])
    network.biases[2] = np.array([[1], [1]])

    result = network.run(np.array([[2], [1]]))

    assert np.array_equal(result, np.array([[9], [9]]))

