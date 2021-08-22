import random
import numpy as np
import pytest

from ann_point.Functions import *
from evolving_classifier.FitnessFunction import *
from ann_point.Functions import *

from neural_network.ChaosNet import efficiency
from utility.Mut_Utility import gaussian_shift


def get_point():
    links = np.array([[0, 0, 1, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0]])
    weights = np.array([[0, 0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 0.5, 0.5, 0.5, -0.5]])
    actFuns = [None, None, Sigmoid(), None, None, None]
    cn = ChaosNet(input_size=2, output_size=3, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                  net_it=2, mutation_radius=0, sqr_mut_prob=0, lin_mut_prob=0, p_mutation_prob=0, c_prob=0, dstr_mut_prob=0)

    return cn

def get_point2():
    links = np.array([[0, 0, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 1, 1, 1, 1, 1],
                      [0, 0, 1, 0, 1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    weights = np.array([[0, 0, 1, 2, 1, 0, 0, 0, 0],
                        [0, 0, -1, 1, -1, 0, 0, 0, 0],
                        [0, 0, 0, 1, -2, 1, 1, 1, -1],
                        [0, 0, 1, 0, 1, 1, 1, 1, -0.5],
                        [0, 0, 1, 1, 0, -2, 0.5, 1, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])
    bias = np.array([[0, 0, 0.5, 1, -1, 0.5, 0.5, -0.5, 0.75]])
    actFuns = [None, None, Sigmoid(), ReLu(), Identity(), Sigmoid(), None, None, None]
    cn = ChaosNet(input_size=2, output_size=4, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                  net_it=4, mutation_radius=0, sqr_mut_prob=0, lin_mut_prob=0, p_mutation_prob=0, c_prob=0, dstr_mut_prob=0)

    return cn

# TODO - S - is test tested?
def get_io():
    inputs = [np.array([[0], [0]]), np.array([[0], [1]]), np.array([[1], [0]]), np.array([[1], [1]])]
    output = [np.array([[1], [0], [0]]), np.array([[0], [1], [0]]), np.array([[0], [1], [0]]), np.array([[0], [0], [1]])]

    inputs.extend([c.copy() for c in inputs])
    inputs.extend([c.copy() for c in inputs])
    output.extend([c.copy() for c in output])
    output.extend([c.copy() for c in output])

    return inputs, output

def get_io2():
    inputs = [np.array([[0.], [0]]), np.array([[0.], [1]]), np.array([[1.], [0]]), np.array([[1.], [1]])]
    output = [np.array([[1.], [0], [0], [0]]), np.array([[0.], [1], [0], [0]]), np.array([[0.], [0], [1], [0]]), np.array([[0.], [0], [0], [1]])]

    inputs.extend([gaussian_shift(c.copy(), np.ones(c.shape), 1, 1) for c in inputs])
    inputs.extend([gaussian_shift(c.copy(), np.ones(c.shape), 1, 1) for c in inputs])
    output.extend([c.copy() for c in output])
    output.extend([c.copy() for c in output])

    return inputs, output

def test_pure_fitness_function():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = CNFF()
    res = ff.compute(point, i, o)

    assert res[0] == pytest.approx(0.19999, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                           [4., 0., 0.]]))


def test_pure_fitness_function2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)


    point = get_point2()
    i, o = get_io2()
    ff = CNFF()
    res = ff.compute(point, i, o)

    assert res[0] == pytest.approx(0.3546063, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))

def test_mixed_loss_1_fun():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = CNFF2(QuadDiff())
    res = ff.compute(point, i, o)

    assert res[0] == pytest.approx(-2.72808896, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                            [4., 0., 0.]]))

def test_mixed_loss_1_fun2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point2()
    i, o = get_io2()
    ff = CNFF2(QuadDiff())
    res = ff.compute(point, i, o)

    assert res[0] == pytest.approx(-2.1653777, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))

def test_mixed_loss_2_fun():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = CNFF3(QuadDiff())
    res = ff.compute(point, i, o)

    assert res[0] == pytest.approx(-2.18247117, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                            [4., 0., 0.]]))

def test_mixed_loss_2_fun2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point2()
    i, o = get_io2()
    ff = CNFF3(QuadDiff())
    res = ff.compute(point, i, o)

    assert res[0] == pytest.approx(-1.39752112, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))

def test_pure_loss_fun():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = CNFF4(QuadDiff())
    res = ff.compute(point, i, o)

    assert res[0] == pytest.approx(-3.4101112, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                            [4., 0., 0.]]))

def test_pure_loss_fun2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point2()
    i, o = get_io2()
    ff = CNFF4(QuadDiff())
    res = ff.compute(point, i, o)

    assert res[0] == pytest.approx(-3.355127, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))


#
# seed = 1001
# random.seed(seed)
# np.random.seed(seed)
# net = get_point()
# i, o = get_io()
#
# test = net.test(i, o)
# print(efficiency(test[0]))
# print(test[0])
#
# test_pure_fitness_function()
#
# seed = 1001
# random.seed(seed)
# np.random.seed(seed)
# net = get_point2()
# i, o = get_io2()
#
# test = net.test(i, o)
# print(efficiency(test[0]))
# print(test[0])
#
# test_pure_fitness_function2()
#
# seed = 1001
# random.seed(seed)
# np.random.seed(seed)
# net = get_point()
# i, o = get_io()
#
# test = net.test(i, o, QuadDiff())
# print(-test[1])
# print(test[0])
#
# test_pure_loss_fun()
#
# seed = 1001
# random.seed(seed)
# np.random.seed(seed)
# net = get_point2()
# i, o = get_io2()
#
# test = net.test(i, o, QuadDiff())
# print(-test[1])
# print(test[0])
#
# test_pure_loss_fun2()
#
#
#
# seed = 1001
# random.seed(seed)
# np.random.seed(seed)
# net = get_point()
# i, o = get_io()
#
# test = net.test(i, o, QuadDiff())
# print((1 - efficiency(test[0])) * -test[1])
# print(test[0])
#
# test_mixed_loss_1_fun()
#
# seed = 1001
# random.seed(seed)
# np.random.seed(seed)
# net = get_point2()
# i, o = get_io2()
#
# test = net.test(i, o, QuadDiff())
# print((1 - efficiency(test[0])) * -test[1])
# print(test[0])
#
# test_mixed_loss_1_fun2()
#
#
# seed = 1001
# random.seed(seed)
# np.random.seed(seed)
# net = get_point()
# i, o = get_io()
#
# test = net.test(i, o, QuadDiff())
# print((1 - efficiency(test[0]))**2 * -test[1])
# print(test[0])
#
# test_mixed_loss_2_fun()
#
# seed = 1001
# random.seed(seed)
# np.random.seed(seed)
# net = get_point2()
# i, o = get_io2()
#
# test = net.test(i, o, QuadDiff())
# print((1 - efficiency(test[0]))**2 * -test[1])
# print(test[0])
#
# test_mixed_loss_2_fun2()







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