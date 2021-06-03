from neural_network.FeedForwardNeuralNetwork import *
import numpy as np
import pytest

def get_io():
    inputs = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]])]
    output = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])]

    return inputs, output

def test_nn_test():
    inputs, output = get_io()
    network = FeedForwardNeuralNetwork(neuronCounts=[2, 4, 3], actFun=[ReLu(), Sigmoid()], lossFun=CrossEntropy(),
                                       learningRate=-3, momCoeffL=-3, batchSize=-2, seed=1010)

    res = network.test(inputs, output)

    assert res[0] == pytest.approx(0.5, abs=1e-3)
    assert res[1] == pytest.approx(0.33333, abs=1e-3)
    assert res[2] == pytest.approx(0.5, abs=1e-3)
    assert np.array_equal(res[3], np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]]))

input, output = get_io()
network = FeedForwardNeuralNetwork(neuronCounts=[2, 4, 3], actFun=[ReLu(), Sigmoid()], lossFun=CrossEntropy(),
                                   learningRate=-3, momCoeffL=-3, batchSize=-2, seed=1010)

# res1 = network.run(input[0])
# res2 = network.run(input[1])
# res3 = network.run(input[2])
# res4 = network.run(input[3])
#
# print(np.argmax(res1))
# print(np.argmax(res2))
# print(np.argmax(res3))
# print(np.argmax(res4))
#
# cm = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]])
# print(accuracy(cm))
# print(average_precision(cm))
# print(average_recall(cm))
# print(efficiency(cm))
#
# test_nn_test()

