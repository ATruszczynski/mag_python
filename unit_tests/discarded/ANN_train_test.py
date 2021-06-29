from neural_network.FeedForwardNeuralNetwork import *
from ann_point.Functions import *
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from statistics import mean, stdev
import pytest

#TODO test generation of counting problem
#TODO create an unbalanced test problem
#TODO test numpy copy
#TODO test deepcopying in general

def test_train_relus():
    network = FeedForwardNeuralNetwork(neuronCounts=[2, 2, 2], actFun=[ReLu(), ReLu()], lossFun=QuadDiff(), learningRate=0, momCoeff=0, batchSize=0, seed=1001)

    inputs = [np.array([[2], [1]])]
    outputs = [np.array([[1], [0]])]

    network.weights[1] = np.array([[1, 1], [1, 1]], dtype=float)
    network.weights[2] = np.array([[1, 1], [1, 1]], dtype=float)
    network.biases[1] = np.array([[1], [1]], dtype=float)
    network.biases[2] = np.array([[1], [1]], dtype=float)

    network.train(inputs, outputs, 1)

    assert len(network.weights) == 3
    assert len(network.biases) == 3
    assert np.array_equal(network.weights[1], np.array([[-33, -16], [-33, -16]]))
    assert np.array_equal(network.weights[2], np.array([[-31, -31], [-35, -35]]))
    assert np.array_equal(network.biases[1], np.array([[-16], [-16]]))
    assert np.array_equal(network.biases[2], np.array([[-7], [-8]]))

    result = network.run(inputs[0])

    assert np.array_equal(result, np.array([[0], [0]]))

def test_nn_determinism():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    X = [x.reshape((4, 1)) for x in X]
    y = one_hot_endode(y)

    perm = list(range(0, len(y)))
    random.shuffle(perm)

    X = [X[i] for i in perm]
    y = [y[i] for i in perm]

    point = AnnPoint([4, 4, 3], [ReLu(), ReLu()], QuadDiff(), -3, -3, -3)

    results = []
    wb_matrixes = []

    n = 10
    for i in range(10):
        network = network_from_point(point, 1001)
        network.train(X, y, 10)
        res = network.test(X, y)
        results.append(res)
        wb_matrix = []
        for i in range(len(network.weights)):
            wb_matrix.append(network.weights[i].copy())
            wb_matrix.append(network.biases[i].copy())
        wb_matrixes.append(wb_matrix)


    for i in range(n):
        for j in range(i + 1, n):
            res1 = results[i]
            res2 = results[j]
            assert len(res1) == len(res2)
            assert res1[0] == res2[0]
            assert res1[1] == res2[1]
            assert res1[2] == res2[2]
            assert np.array_equal(res1[3], res2[3])

            wb1 = wb_matrixes[i]
            wb2 = wb_matrixes[j]

            assert len(wb1) == len(wb2)

            for k in range(len(wb1)):
                assert np.array_equal(wb1[k], wb2[k])

def test_train_batch_counts():
    inputs = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]])]
    output = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])]

    inputs.extend([c.copy() for c in inputs])
    inputs.extend([c.copy() for c in inputs])
    output.extend([c.copy() for c in output])
    output.extend([c.copy() for c in output])

    point = AnnPoint([2, 4, 3], [ReLu(), ReLu()], QuadDiff(), -3, -3, -1)
    network = network_from_point(point, 1001)
    network.train(inputs, output, 3)

    assert len(network.cm_hist) == 6
    #TODO is layering and delayering of point tested?

def test_staggered_run():
    inputs = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]])]
    output = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])]
    inputs.extend([c.copy() for c in inputs])
    inputs.extend([c.copy() for c in inputs])
    output.extend([c.copy() for c in output])
    output.extend([c.copy() for c in output])

    point = AnnPoint([2, 4, 3], [ReLu(), ReLu()], QuadDiff(), -3, -3, -1)

    network = network_from_point(point, 1001)

    ep = 3

    for i in range(ep):
        network.train(inputs, output, 1)

    res1 = network.test(inputs, output)
    w1 = network.weights
    b1 = network.biases

    network = network_from_point(point, 1001)
    network.train(inputs, output, 3)
    res2 = network.test(inputs, output)
    w2 = network.weights
    b2 = network.biases

    assert len(res1) == len(res2)
    assert len(w1) == len(w2)
    assert len(b1) == len(b2)
    assert len(w1) == len(b1)

    assert res1[0] == res2[0]
    assert res1[1] == res2[1]
    assert res1[2] == res2[2]
    assert np.array_equal(res1[3], res2[3])

    for i in range(len(w1)):
        assert np.array_equal(w1[i], w2[i])
        assert np.array_equal(b1[i], b2[i])

#
# def test_nn_staggered_run():
#     point = AnnPoint(inputSize=5, outputSize=6, hiddenLayerCount=1, neuronCount=4, actFun=TanH(), aggrFun=Softmax(), lossFun=CrossEntropy(), learningRate=-4, momCoeff=-4, batchSize=-2)
#
#     network = network_from_point(point, 1001)
#
#     count = 100
#     size = 5
#     x,y = generate_counting_problem(count, size)
#     X,Y = generate_counting_problem(ceil(count/5), size)
#
#     its = 10
#
#     for i in range(its):
#         network.train(x, y, epochs=1)
#
#     test1 = network.test(test_input=X, test_output=Y)
#
#     network = network_from_point(point, 1001)
#
#     network.train(x, y, epochs=its)
#
#     test2 = network.test(test_input=X, test_output=Y)
#
#     assert test1[0] == test2[0]
#     assert test1[1] == test2[1]
#     assert test1[2] == test2[2]
#     assert np.array_equal(test1[3], test2[3])
#
#
#
#
# test_run_relus()
#
#
#
#