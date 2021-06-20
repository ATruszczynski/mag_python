import random
import numpy as np
import pytest

from ann_point.AnnPoint import AnnPoint
from ann_point.Functions import *
from evolving_classifier.FitnessFunction import *

#TODO Softmax makes no sense on single neuron

def get_point():
    links = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    weights = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 1, 1, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 0.5, 0.5, 0.5, -0.5]])
    actFuns = [None, None, Sigmoid(), None, None]
    cn = ChaosNet(input_size=2, output_size=3, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax())

    return cn

# TODO is test tested?
def get_io():
    inputs = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]])]
    output = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])]

    inputs.extend([c.copy() for c in inputs])
    inputs.extend([c.copy() for c in inputs])
    output.extend([c.copy() for c in output])
    output.extend([c.copy() for c in output])

    return inputs, output

#TODO more tests of fitness funcs

#TODO pytest finds 2 tests here

def test_pure_fitness_function():
    point = get_point()
    i, o = get_io()
    ff = CNFF()
    res = ff.compute(point, i, o, 1001)

    assert res[0] == pytest.approx(0.22222, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                           [4., 0., 0.]]))

# def test_progress_ff():
#     point = get_point()
#     i, o = get_io()
#     ff = ProgressFF(3)
#     res = ff.compute(point, i, o, 1001)
#
#     assert res[0] == pytest.approx(0.2222 * 0.375, abs=1e-3)
#     assert np.array_equal(res[1], np.array([[4., 0., 0.],[8., 0., 0.],[4., 0., 0.]]))
#
# def test_progress2_ff():
#     point = get_point()
#     i, o = get_io()
#     ff = ProgressFF2(3)
#     res = ff.compute(point, i, o, 1001)
#
#     assert res[0] == pytest.approx(0.2222 * 0.68899, abs=1e-3)
#     assert np.array_equal(res[1], np.array([[4., 0., 0.],[8., 0., 0.],[4., 0., 0.]]))
#
# def test_pure_progress_ff():
#     point = get_point()
#     i, o = get_io()
#     ff = PureProgressFF(3)
#     res = ff.compute(point, i, o, 1001)
#
#     assert res[0] == pytest.approx(0.375, abs=1e-3)
#     assert np.array_equal(res[1], np.array([[4., 0., 0.],[8., 0., 0.],[4., 0., 0.]]))


seed = 1001
random.seed(seed)
np.random.seed(seed)
net = get_point()
i, o = get_io()

test = net.test(i, o)
print(efficiency(test[3]))
print(test[3])


#
# network = network_from_point(point, seed)
# network.train(i, o, 1)
# test1 = network.test(i, o)
#
# network.train(i, o, 1)
# test2 = network.test(i, o)
#
# network.train(i, o, 1)
# test3 = network.test(i, o)
#
# eff1 = efficiency(test1[3])
# eff2 = efficiency(test2[3])
# eff3 = efficiency(test3[3])
#
#
#
# print(eff1)
# print(eff2)
# print(eff3)
#
# print(test2[:3])
# print(test2[3])
# print(efficiency(test2[3]))
#
# y = np.array([eff1, eff2, eff3])
# x = np.array([0, 1, 2])
#
# x = x.reshape((-1, 1))
# y = y.reshape((-1, 1))
#
# reg = LinearRegression().fit(x, y)
# slope = reg.coef_
#
# print(punishment_function(slope))
# print(test3[3])
#
# y = np.array([efficiency(network.cm_hist[i]) for i in range(len(network.cm_hist))]).reshape((-1, 1))
# x = np.array(list(range(len(network.cm_hist)))).reshape((-1, 1))
#
# reg = LinearRegression().fit(x, y)
# slope = reg.coef_
# print(punishment_function(slope))
#
#
#
# # test_pure_fitness_function()
# test_progress_ff()
# test_pure_progress_ff()