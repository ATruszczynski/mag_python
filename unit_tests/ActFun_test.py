import pytest
from ann_point.Functions import *

def test_relu():
    arg = np.array([[9], [1], [0], [-1]])
    exp = np.array([[9], [1], [0], [0]])

    assert np.array_equal(exp, ReLu().compute(arg))

def test_relu_der():
    arg = np.array([[9], [0], [0.25], [-1]])
    exp = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])

    assert np.array_equal(exp, ReLu().computeDer(arg))

def test_sigma():
    arg = np.array([[0], [-1], [2]])

    sigma = Sigmoid()

    assert np.all(np.isclose(sigma.compute(arg), np.array([[0.5], [0.268941421], [0.880797077]]), atol=1e-6))

def test_sigma_der():
    arg = np.array([[0], [-1], [2]])

    sigma = Sigmoid()

    assert np.all(np.isclose(sigma.computeDer(arg), np.array([[0.25, 0, 0], [0, 0.196611933, 0], [0, 0, 0.104993585]]), atol=1e-6))

def test_softmax():
    arg = np.array([[1], [2], [0.5]])

    sm = Softmax()

    assert np.all(np.isclose(sm.compute(arg), np.array([[0.231223898], [0.628531719], [0.140244383]]), atol=1e-5))

def test_softmax_der():
    arg = np.array([[1], [2], [0.5]])

    sm = Softmax()

    res = sm.computeDer(arg)

    assert np.all(np.isclose(res, np.array([[0.177759407, -0.145331554, -0.032427853],
                                                        [-0.145331554, 0.233479597, -0.088148043],
                                                        [-0.032427853, -0.088148043, 0.120575896]]), atol=1e-5))

def test_tanh():
    arg = np.array([[1], [-2], [0.5]])
    tanh = TanH()
    res = tanh.compute(arg)

    assert np.all(np.isclose(res, np.array([[0.761594156], [-0.96402758], [0.462117157]])))

def test_tanh_der():
    arg = np.array([[1], [-2], [0.5]])
    tanh = TanH()
    res = tanh.computeDer(arg)

    assert np.all(np.isclose(res, np.array([[0.419974342, 0, 0],
                                            [0, 0.070650825, 0],
                                            [0, 0, 0.786447733]]), atol=1e-5))

