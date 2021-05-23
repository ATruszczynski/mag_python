# from neural_network.FeedForwardNeuralNetwork import *
# from ann_point.Functions import *
# import numpy as np
# from sklearn import datasets
# from sklearn.preprocessing import OneHotEncoder
# from statistics import mean, stdev
# import pytest
#
#
# def test_run_relus():
#     network = FeedForwardNeuralNetwork(inputSize=2, outputSize=2, hiddenLayerCount=1, neuronCount=1, actFun=ReLu(),
#                                        aggrFun=ReLu(), lossFun=QuadDiff(), learningRate=0, momCoeffL=0, batchSize=0, seed=1001)
#
#     inputs = [np.array([[2], [1]])]
#     outputs = [np.array([[1], [0]])]
#
#     network.weights[1] = np.array([[1, 1], [1, 1]], dtype=float)
#     network.weights[2] = np.array([[1, 1], [1, 1]], dtype=float)
#     network.biases[1] = np.array([[1], [1]], dtype=float)
#     network.biases[2] = np.array([[1], [1]], dtype=float)
#
#     network.train(inputs, outputs, 1)
#
#     assert np.array_equal(network.weights[1], np.array([[-33, -16], [-33, -16]]))
#     assert np.array_equal(network.weights[2], np.array([[-31, -31], [-35, -35]]))
#     assert np.array_equal(network.biases[1], np.array([[-16], [-16]]))
#     assert np.array_equal(network.biases[2], np.array([[-7], [-8]]))
#
#     result = network.run(inputs[0])
#
#     assert np.array_equal(result, np.array([[0], [0]]))
#
# def test_determinism():
#     iris = datasets.load_iris()
#     X = iris.data
#     y = iris.target
#
#     X = [x.reshape((4, 1)) for x in X]
#     y = one_hot_endode(y)
#
#     perm = list(range(0, len(y)))
#     random.shuffle(perm)
#
#     X = [X[i] for i in perm]
#     y = [y[i] for i in perm]
#
#     point = AnnPoint(4, 3, 2, 4, ReLu(), ReLu(), QuadDiff(), -3, -3, -3)
#     # TODO nie sprawdza wag itp.
#     results = []
#
#     for i in range(10):
#         network = network_from_point(point, 1001)
#         network.train(X, y, 10)
#         res = network.test(X, y)[0:3]
#         results.append(mean(res))
#
#     sd = stdev(results)
#
#     assert sd == pytest.approx(0)
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
# # TODO test multiple runs vs 1 longer run
#
# test_run_relus()
#
#
#
#
