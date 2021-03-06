import random
import numpy as np
import pytest

from ann_point.Functions import *
from evolving_classifier.FitnessFunction import *
from ann_point.Functions import *

from neural_network.LsmNetwork import efficiency
from utility.Mut_Utility import gaussian_shift, get_default_hrange_ga
from utility.Utility import generate_population


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
    cn = LsmNetwork(input_size=2, output_size=3, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                    net_it=2, mutation_radius=0, swap_prob=0, multi=0, p_prob=0, c_prob=0, p_rad=0)

    return cn

def get_point_3():
    random.seed(1001)
    hrange = get_default_hrange_ga()
    cn = generate_population(hrange=hrange, count=1, input_size=2, output_size=2)[0]

    return cn

def get_point_4():
    random.seed(1002)
    hrange = get_default_hrange_ga()
    cn = generate_population(hrange=hrange, count=1, input_size=2, output_size=2)[0]

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
    cn = LsmNetwork(input_size=2, output_size=4, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                    net_it=4, mutation_radius=0, swap_prob=0, multi=0, p_prob=0, c_prob=0, p_rad=0)

    return cn

def get_point3():
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
    cn = LsmNetwork(input_size=2, output_size=2, links=links, weights=weights, biases=bias, actFuns=actFuns, aggrFun=Softmax(),
                    net_it=4, mutation_radius=0, swap_prob=0, multi=0, p_prob=0, c_prob=0, p_rad=0)

    return cn

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

def get_io3():
    inputs = [np.array([[0.], [0]]), np.array([[0.], [1]]), np.array([[1.], [0]]), np.array([[1.], [1]])]
    output = [np.array([[1.], [0]]), np.array([[0.], [1]]), np.array([[1.], [0]]), np.array([[0.], [1]])]

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

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(0.19999, abs=1e-3)
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

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(0.3546063, abs=1e-3)
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

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(-2.72808896/16, abs=1e-3)
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

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(-2.1653777/16, abs=1e-3)
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

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(-2.18247117/16, abs=1e-3)
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

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(-1.39752112/16, abs=1e-3)
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

    assert len(res[0]) == 2
    assert res[0][0] == pytest.approx(-3.4101112/16, abs=1e-3)
    assert res[0][1] == pytest.approx(-6, abs=1e-3)
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

    assert len(res[0]) == 2
    assert res[0][0] == pytest.approx(-3.355127/16, abs=1e-3)
    assert res[0][1] == pytest.approx(-9, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))
def test_meff_loss_1():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = CNFF5()
    res = ff.compute(point, i, o)

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(0.0999999, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                            [4., 0., 0.]]))

def test_meff_loss_2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point2()
    i, o = get_io2()
    ff = CNFF5()
    res = ff.compute(point, i, o)

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(0.177303165, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))
def test_mixed_meff_loss_1():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = MIXFF(QuadDiff())
    res = ff.compute(point, i, o)

    assert len(res[0]) == 3
    assert res[0][0] == pytest.approx(-3.410111209/16, abs=1e-3)
    assert res[0][1] == pytest.approx(-0.21313195, abs=1e-3)
    assert res[0][2] == pytest.approx(-6, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                            [4., 0., 0.]]))

def test_mixed_meff_loss_2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point2()
    i, o = get_io2()
    ff = MIXFF(QuadDiff())
    res = ff.compute(point, i, o)

    assert len(res[0]) == 3
    assert res[0][0] == pytest.approx(-3.355127113/16, abs=1e-3)
    assert res[0][1] == pytest.approx(-0.20969544, abs=1e-3)
    assert res[0][2] == pytest.approx(-9, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))


def test_avmin_loss_1():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = AVMAX(QuadDiff())
    res = ff.compute(point, i, o)

    assert len(res[0]) == 2
    assert res[0][0] == pytest.approx(-0.2232926431627285, abs=1e-3)
    assert res[0][1] == pytest.approx(-6, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                            [4., 0., 0.]]))

def test_avmin_loss_2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point2()
    i, o = get_io2()
    ff = AVMAX(QuadDiff())
    res = ff.compute(point, i, o)

    assert len(res[0]) == 2
    assert res[0][0] == pytest.approx(-0.229847722, abs=1e-3)
    assert res[0][1] == pytest.approx(-9, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))


def test_MEFF_loss_1():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = MEFF()
    res = ff.compute(point, i, o)

    assert len(res[0]) == 2
    assert res[0][0] == pytest.approx(0.0, abs=1e-3)
    assert res[0][1] == pytest.approx(0.13333333, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                            [4., 0., 0.]]))

def test_MEFF_loss_2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point2()
    i, o = get_io2()
    ff = MEFF()
    res = ff.compute(point, i, o)

    assert len(res[0]) == 2
    assert res[0][0] == pytest.approx(0.0, abs=1e-3)
    assert res[0][1] == pytest.approx(0.30681818, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))


def test_FFPUNEFF_loss_1():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point_3()
    i, o = get_io3()
    ff = FFPUNEFF()
    res = ff.compute(point, i, o)

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(-40.0, abs=1e-3)
    assert np.array_equal(res[1], np.array([[8., 0.],
                                            [8., 0.]]))

def test_FFPUNEFF_loss_2():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point_4()
    i, o = get_io3()
    ff = FFPUNEFF()
    res = ff.compute(point, i, o)

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(-40.0, abs=1e-3)
    assert np.array_equal(res[1], np.array([[8., 0.],
                                            [8., 0.]]))




def test_FFPUNQD_loss_1():
    seed = 1001
    random.seed(seed)
    np.random.seed(seed)

    point = get_point()
    i, o = get_io()
    ff = FFPUNQD(QuadDiff())
    res = ff.compute(point, i, o)

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(-8.52527802, abs=1e-3)
    assert np.array_equal(res[1], np.array([[4., 0., 0.],
                                            [8., 0., 0.],
                                            [4., 0., 0.]]))

def test_FFPUNQD_loss_2():
    seed = 1001
    random.seed(seed)

    np.random.seed(seed)
    point = get_point2()
    i, o = get_io2()
    ff = FFPUNQD(QuadDiff())
    res = ff.compute(point, i, o)

    assert len(res[0]) == 1
    assert res[0][0] == pytest.approx(-3.14543166, abs=1e-3)
    assert np.array_equal(res[1], np.array([[3., 0., 1., 0.],
                                            [3., 0., 1., 0.],
                                            [0., 0., 4., 0.],
                                            [2., 1., 1., 0.]]))

