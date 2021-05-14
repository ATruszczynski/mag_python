from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from statistics import mean, stdev
import pytest


def test_run_relus():
    network = FeedForwardNeuralNetwork(inputSize=2, outputSize=2, hiddenLayerCount=1, neuronCount=1, actFun=ReLu(),
                                       aggrFun=ReLu(), lossFun=QuadDiff(), learningRate=0, momCoeffL=0, seed=1001)

    inputs = [np.array([[2], [1]])]
    outputs = [np.array([[1], [0]])]

    network.weights[1] = np.array([[1, 1], [1, 1]], dtype=float)
    network.weights[2] = np.array([[1, 1], [1, 1]], dtype=float)
    network.biases[1] = np.array([[1], [1]], dtype=float)
    network.biases[2] = np.array([[1], [1]], dtype=float)

    network.train(inputs, outputs, 1, 1)

    assert np.array_equal(network.weights[1], np.array([[-67, -33], [-67, -33]]))
    assert np.array_equal(network.weights[2], np.array([[-63, -63], [-71, -71]]))
    assert np.array_equal(network.biases[1], np.array([[-33], [-33]]))
    assert np.array_equal(network.biases[2], np.array([[-15], [-17]]))

    result = network.run(inputs[0])

    assert np.array_equal(result, np.array([[0], [0]]))

def test_determinism():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = [x.reshape((4, 1)) for x in X]
    y = one_hot_endode(y)

    perm = list(range(0, len(y)))
    random.shuffle(perm)

    X = [X[i] for i in perm]
    y = [y[i] for i in perm]

    point = AnnPoint(4, 3, 2, 4, ReLu(), ReLu(), QuadDiff(), -3, -3)

    results = []

    for i in range(10):
        network = network_from_point(point, 1001)
        network.train(X, y, 10, 10)
        res = network.test(X, y)[0:3]
        results.append(mean(res))

    sd = stdev(results)

    assert sd == pytest.approx(0)

test_determinism()




